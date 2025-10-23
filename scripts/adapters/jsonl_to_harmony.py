#!/usr/bin/env python3
"""Convert diarised transcript JSONL files into Harmony fine-tuning records.

The script groups consecutive speaker turns, maps them to Harmony roles, and
emits train/val JSONL shards that follow the three-message Harmony schema:
system -> user -> assistant (analysis/final channels). It keeps dependencies
lightweight so it can run inside the repository without extra setup.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


try:  # pragma: no cover - optional dependency
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_TOKENIZER.encode(text))

except Exception:  # pragma: no cover - fallback path

    def count_tokens(text: str) -> int:
        return max(0, len(text) // 4)


@dataclass
class Segment:
    """Single diarised utterance."""

    episode_id: str
    speaker: str
    text: str
    start: Optional[float]
    end: Optional[float]
    order: int
    source_path: Path


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_segments(path: Path, min_chars: int) -> List[Segment]:
    segments: List[Segment] = []
    with path.open("r", encoding="utf-8") as handle:
        for order, line in enumerate(handle):
            if not line.strip():
                continue
            data = json.loads(line)
            text = (data.get("text") or "").strip()
            if len(text) < min_chars:
                continue
            segments.append(
                Segment(
                    episode_id=str(data.get("episode_id") or path.stem),
                    speaker=str(data.get("speaker") or "unknown"),
                    text=text,
                    start=parse_float(data.get("start")),
                    end=parse_float(data.get("end")),
                    order=order,
                    source_path=path,
                )
            )
    segments.sort(key=lambda seg: (seg.start if seg.start is not None else float("inf"), seg.order))
    return segments


def merge_consecutive(segments: Sequence[Segment], merge_gap: float) -> List[Segment]:
    if not segments:
        return []

    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        same_speaker = seg.speaker == prev.speaker
        gap_ok = False
        if prev.end is not None and seg.start is not None and merge_gap >= 0:
            gap_ok = seg.start - prev.end <= merge_gap
        elif prev.end is None or seg.start is None:
            gap_ok = True

        if same_speaker and gap_ok:
            prev.text = f"{prev.text} {seg.text}".strip()
            prev.end = seg.end if seg.end is not None else prev.end
        else:
            merged.append(seg)
    return merged


def window_segments(segments: Sequence[Segment], max_turns: int) -> List[List[Segment]]:
    if max_turns <= 0:
        return [list(segments)]
    return [list(segments[i : i + max_turns]) for i in range(0, len(segments), max_turns)]


def ordered_speakers(segments: Iterable[Segment]) -> List[str]:
    seen: List[str] = []
    for seg in segments:
        if seg.speaker not in seen:
            seen.append(seg.speaker)
    return seen


def build_system_prompt(episode_id: str, speakers: List[str], mode: str) -> str:
    speaker_str = ", ".join(speakers) if speakers else "unknown speaker"
    return (
        f"Transcript mode: {mode}. Episode: {episode_id}. "
        f"Speakers observed: {speaker_str}. Preserve the diarised ordering and attribution."
    )


def assistant_payload(speaker: str, text: str) -> Dict[str, str]:
    analysis = f"Direct transcript turn spoken by {speaker}. Reasoning not available."
    return {"role": "assistant", "analysis": analysis, "final": text}


def conversation_messages(
    segments: Sequence[Segment], speaker_roles: Dict[str, str], system_prompt: str
) -> List[Dict[str, object]]:
    messages: List[Dict[str, object]] = [{"role": "system", "content": system_prompt}]
    last_role: Optional[str] = None

    for seg in segments:
        role = speaker_roles.get(seg.speaker, "user")
        if role == "assistant":
            messages.append(assistant_payload(seg.speaker, seg.text))
            last_role = role
            continue

        content = f"{seg.speaker}: {seg.text}"
        if last_role == "user":
            messages[-1]["content"] += f"\n{content}"  # type: ignore[index]
        else:
            messages.append({"role": "user", "content": content})
        last_role = "user"

    has_assistant = any(msg.get("role") == "assistant" for msg in messages)
    if not has_assistant:
        messages.append(
            assistant_payload("unknown", "No assistant turns detected in this window; placeholder response.")
        )
    return messages


def monologue_messages(
    segments: Sequence[Segment], system_prompt: str, note: str
) -> List[Dict[str, object]]:
    content_lines = [f"{seg.speaker}: {seg.text}" for seg in segments]
    transcript_text = "\n".join(content_lines)
    analysis = f"Placeholder analysis for multi-speaker window. {note}"
    return [
        {"role": "system", "content": f"{system_prompt} {note}"},
        {"role": "user", "content": transcript_text},
        {"role": "assistant", "analysis": analysis, "final": transcript_text},
    ]


def build_records(
    segments: Sequence[Segment],
    mode: str,
    max_turns: int,
) -> List[Dict[str, object]]:
    if not segments:
        return []

    speakers = ordered_speakers(segments)
    system_prompt = build_system_prompt(segments[0].episode_id, speakers, mode)
    windows = window_segments(segments, max_turns)
    records: List[Dict[str, object]] = []

    for window_index, window in enumerate(windows):
        speaker_roles: Dict[str, str] = {}
        if mode == "conversation" and len(speakers) >= 2:
            speaker_roles[speakers[0]] = "user"
            speaker_roles[speakers[1]] = "assistant"

        if mode == "conversation" and "assistant" in speaker_roles.values():
            messages = conversation_messages(window, speaker_roles, system_prompt)
            strategy = "two_speaker"
        else:
            note = "This window contains three or more unique speakers, so the transcript is flattened into a single turn."
            messages = monologue_messages(window, system_prompt, note)
            strategy = "flattened"

        user_contents = [msg["content"] for msg in messages if msg.get("role") == "user"]
        user_token_counts = [count_tokens(content) for content in user_contents]
        assistant_finals = [msg.get("final", "") for msg in messages if msg.get("role") == "assistant"]
        assistant_token_counts = [count_tokens(text) for text in assistant_finals]

        metadata = {
            "episode_id": window[0].episode_id,
            "source_path": str(window[0].source_path),
            "window_index": window_index,
            "windows_total": len(windows),
            "speaker_roles": speaker_roles,
            "speakers": speakers,
            "mode": mode,
            "strategy": strategy,
            "token_estimates": {
                "user_total": sum(user_token_counts),
                "user_max": max(user_token_counts) if user_token_counts else 0,
                "assistant_total": sum(assistant_token_counts),
                "assistant_max": max(assistant_token_counts) if assistant_token_counts else 0,
            },
        }

        records.append({"messages": messages, "metadata": metadata})

    return records


def split_train_val(records: List[Dict[str, object]], train_split: float, seed: int) -> Tuple[List, List]:
    rng = random.Random(seed)
    rng.shuffle(records)
    split_index = int(len(records) * train_split)
    if split_index == len(records) and records:
        split_index = len(records) - 1
    return records[:split_index], records[split_index:]


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_files(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    matched_paths = sorted(Path(p) for pattern in args.input_glob for p in glob.glob(pattern))
    if not matched_paths:
        raise FileNotFoundError(f"No files matched input glob(s): {args.input_glob}")

    all_records: List[Dict[str, object]] = []
    for path in matched_paths:
        segments = load_segments(path, args.min_chars)
        merged = merge_consecutive(segments, args.merge_gaps_sec)
        all_records.extend(build_records(merged, args.mode, args.max_turns))

    return split_train_val(all_records, args.train_split, args.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert diarised transcripts into Harmony JSONL shards.")
    parser.add_argument("--input_glob", nargs="+", required=True, help="Glob(s) pointing to transcript JSONL files.")
    parser.add_argument("--out_dir", type=Path, required=True, help="Output directory for Harmony shards.")
    parser.add_argument("--mode", choices=["conversation", "singleturn"], default="conversation")
    parser.add_argument("--max_turns", type=int, default=20, help="Maximum diarised turns per Harmony record.")
    parser.add_argument("--min_chars", type=int, default=20, help="Discard utterances shorter than this length.")
    parser.add_argument("--merge_gaps_sec", type=float, default=1.0, help="Merge consecutive turns within this gap.")
    parser.add_argument("--train_split", type=float, default=0.95, help="Fraction of records in train.jsonl.")
    parser.add_argument("--seed", type=int, default=7, help="Deterministic shuffle seed.")
    args = parser.parse_args()

    train_records, val_records = process_files(args)

    write_jsonl(args.out_dir / "train.jsonl", train_records)
    write_jsonl(args.out_dir / "val.jsonl", val_records)

    print(f"[adapter] wrote {len(train_records)} train and {len(val_records)} val Harmony records to {args.out_dir}")


if __name__ == "__main__":
    main()

