#!/usr/bin/env python3
"""Validate Harmony JSONL datasets produced by the transcript adapter."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


try:  # pragma: no cover - optional dependency
    import tiktoken

    _TOKENIZER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_TOKENIZER.encode(text))

except Exception:  # pragma: no cover - fallback path

    def count_tokens(text: str) -> int:
        return max(0, len(text) // 4)


def load_jsonl(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:  # pragma: no cover - invalid line
                raise ValueError(f"{path}:{line_no} is not valid JSON ({exc})") from exc
    return records


def ensure_text(value: object, path: str) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return [f"{path} must be a non-empty string"]
    return []


def validate_entry(
    entry: Dict,
    idx: int,
    max_user_tokens: int,
    max_final_tokens: int,
) -> Tuple[List[str], int, int]:
    errors: List[str] = []
    messages = entry.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        return [f"[entry {idx}] messages must be a list with >=3 items"], 0, 0

    system_msg = messages[0]
    if system_msg.get("role") != "system":
        errors.append(f"[entry {idx}] first message must have role=system")
    errors.extend(ensure_text(system_msg.get("content"), f"[entry {idx}] system.content"))

    user_msg = next((msg for msg in messages if msg.get("role") == "user"), None)
    if user_msg is None:
        errors.append(f"[entry {idx}] missing user message")
        user_tokens = 0
    else:
        errors.extend(ensure_text(user_msg.get("content"), f"[entry {idx}] user.content"))
        user_tokens = count_tokens(user_msg.get("content", ""))
        if user_tokens > max_user_tokens:
            errors.append(
                f"[entry {idx}] user.content exceeds {max_user_tokens} tokens (got {user_tokens})"
            )

    assistant_msgs = [msg for msg in messages if msg.get("role") == "assistant"]
    if not assistant_msgs:
        errors.append(f"[entry {idx}] missing assistant message")
        final_tokens = 0
    else:
        final_tokens = 0
        for pos, assistant in enumerate(assistant_msgs, 1):
            errors.extend(
                ensure_text(
                    assistant.get("analysis"),
                    f"[entry {idx}] assistant[{pos}].analysis",
                )
            )
            analysis_tokens = count_tokens(assistant.get("analysis", ""))
            if analysis_tokens > max_final_tokens:
                errors.append(
                    f"[entry {idx}] assistant[{pos}].analysis exceeds {max_final_tokens} tokens "
                    f"(got {analysis_tokens})"
                )

            errors.extend(
                ensure_text(
                    assistant.get("final"),
                    f"[entry {idx}] assistant[{pos}].final",
                )
            )
            final_tokens = count_tokens(assistant.get("final", ""))
            if final_tokens > max_final_tokens:
                errors.append(
                    f"[entry {idx}] assistant[{pos}].final exceeds {max_final_tokens} tokens "
                    f"(got {final_tokens})"
                )

    metadata = entry.get("metadata", {})
    if metadata and not isinstance(metadata, dict):
        errors.append(f"[entry {idx}] metadata must be an object when provided")

    return errors, user_tokens, final_tokens


def percentile(values: Iterable[int], pct: float) -> float:
    seq = sorted(values)
    if not seq:
        return 0.0
    rank = max(0, min(len(seq) - 1, int(round((len(seq) - 1) * pct))))
    return float(seq[rank])


def summarise_tokens(values: List[int]) -> Dict[str, float]:
    if not values:
        return {"avg": 0.0, "p95": 0.0}
    return {
        "avg": statistics.fmean(values),
        "p95": percentile(values, 0.95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Harmony JSONL inputs.")
    parser.add_argument("--input", action="append", required=True, help="Path to JSONL file(s) to validate.")
    parser.add_argument("--max_user_tokens", type=int, default=2000, help="Token budget for user messages.")
    parser.add_argument("--max_final_tokens", type=int, default=400, help="Token budget for assistant analysis/final.")
    args = parser.parse_args()

    overall_errors: List[str] = []

    for input_path in args.input:
        path = Path(input_path)
        records = load_jsonl(path)
        user_tokens: List[int] = []
        final_tokens: List[int] = []
        file_errors: List[str] = []

        for idx, entry in enumerate(records, 1):
            errors, u_tokens, f_tokens = validate_entry(
                entry,
                idx,
                args.max_user_tokens,
                args.max_final_tokens,
            )
            if errors:
                file_errors.extend(f"{path}: {err}" for err in errors)
            if u_tokens:
                user_tokens.append(u_tokens)
            if f_tokens:
                final_tokens.append(f_tokens)

        overall_errors.extend(file_errors)
        stats_user = summarise_tokens(user_tokens)
        stats_final = summarise_tokens(final_tokens)
        print(
            f"[validator] {path} -> {len(records)} records | "
            f"user avg={stats_user['avg']:.1f} p95={stats_user['p95']:.1f} | "
            f"assistant-final avg={stats_final['avg']:.1f} p95={stats_final['p95']:.1f}"
        )

    if overall_errors:
        preview = "\n".join(overall_errors[:20])
        raise SystemExit(f"Harmony validation failed:\n{preview}")

    print("[validator] all inputs passed schema and token checks.")


if __name__ == "__main__":
    main()
