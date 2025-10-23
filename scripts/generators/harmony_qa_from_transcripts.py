#!/usr/bin/env python3
"""Generate Harmony-formatted Q/A data from diarised transcripts."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import math
import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple
from collections import OrderedDict

try:  # Optional dependency for token counting
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENCODER.encode(text))

except Exception:  # pragma: no cover

    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)

try:
    import orjson
except ImportError:  # pragma: no cover
    orjson = None  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI  # type: ignore
from scripts.gates.span_matcher import find_verbatim_span
if "llm.openai_client" in sys.modules:
    llm_questionify_client = sys.modules["llm.openai_client"]  # type: ignore[assignment]
else:
    try:
        from scripts.llm import openai_client as llm_questionify_client
        sys.modules["llm.openai_client"] = llm_questionify_client
    except ModuleNotFoundError:  # pragma: no cover - fallback for direct test imports
        from llm import openai_client as llm_questionify_client  # type: ignore
from scripts.questions.miner import last_non_answer_snippet, mine_local_question
from scripts.scorers.pairfit import (
    get_cross_encoder as _pairfit_get_cross_encoder,
    prepare_cross_encoder,
    score_pairs_ce,
)


@dataclass
class Segment:
    episode_id: str
    start: float
    end: float
    speaker: str
    text: str
    order: int
    source_path: Path


@dataclass
class ContextWindow:
    episode_id: str
    segments: List[Segment]
    anchor_index: int
    window_id: str


@dataclass(frozen=True)
class QuestionifyConfig:
    model: Optional[str]
    temperature: float
    max_output_tokens: int
    enabled: bool = True


@dataclass(frozen=True)
class PairFitConfig:
    model: Optional[str]
    temperature: float
    max_output_tokens: int
    enabled: bool = True


@dataclass(frozen=True)
class AnswerTrimConfig:
    model: Optional[str]
    temperature: float
    max_output_tokens: int
    threshold_tokens: int
    enabled: bool = True


@dataclass(frozen=True)
class SpeakerCheckConfig:
    model: Optional[str]
    temperature: float
    max_output_tokens: int
    enabled: bool = True


@dataclass(frozen=True)
class QCAuditConfig:
    model: Optional[str]
    temperature: float
    max_output_tokens: int
    sample_rate: float
    min_samples: int
    pass_threshold: float
    seed: int
    output_path: Optional[Path]
    enabled: bool = True


@dataclass(frozen=True)
class CompositionConfig:
    variant_b_ratio: float = 0.0
    connector_max_sentences: int = 2
    paraphrase_char_cap: int = 300
    variant_seed: int = 42


MAX_LOCAL_QUESTION_TURNS = 4
QUESTIONIFY_CACHE_CAPACITY_DEFAULT = 4096
QUESTIONIFY_MIN_LENGTH = 8
QUESTIONIFY_MAX_LENGTH = 220
QUESTIONIFY_BLOCKED_PREFIXES = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "which",
    "do",
    "does",
    "did",
    "can",
    "will",
    "should",
)

_tbl = str.maketrans("", "", string.punctuation)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().translate(_tbl)).strip()


def answer_present(src: str, ans: str, min_ratio: float = 0.90) -> bool:
    ns, na = _norm(src), _norm(ans)
    if not na:
        return False
    if na in ns:
        return True
    from difflib import SequenceMatcher

    return SequenceMatcher(None, ns, na).ratio() >= min_ratio


def ensure_openai_api_key() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return

    search_paths = [
        Path(".env"),
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for env_path in search_paths:
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENAI_API_KEY":
                    os.environ.setdefault("OPENAI_API_KEY", value.strip().strip('"').strip("'"))
                    return

    raise RuntimeError("OPENAI_API_KEY not found. Set it via environment or .env.")


def parse_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def load_segments(path: Path) -> List[Segment]:
    segments: List[Segment] = []
    with path.open("r", encoding="utf-8") as handle:
        for order, line in enumerate(handle):
            if not line.strip():
                continue
            data = json.loads(line)
            segments.append(
                Segment(
                    episode_id=str(data.get("episode_id") or path.stem),
                    start=parse_float(data.get("start")),
                    end=parse_float(data.get("end")),
                    speaker=str(data.get("speaker") or "Unknown"),
                    text=(data.get("text") or "").strip(),
                    order=order,
                    source_path=path,
                )
            )
    return segments


def format_timestamp(seconds: float) -> str:
    if math.isnan(seconds) or seconds < 0:
        return "?:??"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def format_segment(seg: Segment) -> str:
    return f"[{format_timestamp(seg.start)}–{format_timestamp(seg.end)}] {seg.speaker}: {seg.text}"


def build_windows(
    segments: Sequence[Segment],
    answer_speakers: Sequence[str],
    context_turns: int,
    max_context_tokens: int,
    window_stride: int,
) -> List[ContextWindow]:
    whitelist = {speaker.lower() for speaker in answer_speakers}
    windows: List[ContextWindow] = []
    stride = max(1, window_stride)

    anchor_indices = [idx for idx, seg in enumerate(segments) if seg.speaker.lower() in whitelist]

    for pos in range(0, len(anchor_indices), stride):
        idx = anchor_indices[pos]
        seg = segments[idx]

        start = max(0, idx - context_turns)
        end = min(len(segments), idx + context_turns + 1)
        window_segments = list(segments[start:end])

        while count_tokens("\n".join(format_segment(s) for s in window_segments)) > max_context_tokens:
            if len(window_segments) <= 1:
                break
            if (idx - start) >= (end - idx - 1):
                window_segments.pop(0)
                start += 1
            else:
                window_segments.pop()
                end -= 1

        anchor_index = min(len(window_segments) - 1, max(0, idx - start))
        window_id = f"{seg.episode_id}|{seg.order}"
        windows.append(
            ContextWindow(
                episode_id=seg.episode_id,
                segments=window_segments,
                anchor_index=anchor_index,
                window_id=window_id,
            )
        )
    return windows


def load_transcripts(patterns: Sequence[str]) -> List[Segment]:
    matched_paths = sorted(Path(p) for pattern in patterns for p in glob.glob(pattern))
    if not matched_paths:
        raise FileNotFoundError(f"No files matched input_glob: {patterns}")

    segments: List[Segment] = []
    for path in matched_paths:
        segments.extend(load_segments(path))
    return segments


def normalise(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return " ".join(text.split()).lower()


def merge_contiguous_segments(segments: List[Segment], max_gap: float = 1.0, max_length: int = 1200) -> List[Segment]:
    """Merge adjacent segments if gap ≤ max_gap and total length ≤ max_length."""
    if not segments:
        return segments

    merged = []
    current_group = [segments[0]]

    for i in range(1, len(segments)):
        current_seg = segments[i]
        last_seg = current_group[-1]

        # Calculate gap between segments
        gap = current_seg.start - last_seg.end

        # Check if segments are contiguous (gap <= max_gap) - merge at ≤1.0s per requirements
        if gap <= max_gap:
            # Check if adding this segment would exceed max_length
            combined_text = " ".join(seg.text for seg in current_group) + " " + current_seg.text
            if len(combined_text.strip()) <= max_length:
                current_group.append(current_seg)
            else:
                # Would exceed length, save current group and start new one
                merged.append(_combine_segments(current_group))
                current_group = [current_seg]
        else:
            # Gap too large, save current group and start new one
            merged.append(_combine_segments(current_group))
            current_group = [current_seg]

    # Add the last group
    if current_group:
        merged.append(_combine_segments(current_group))

    return merged

def _combine_segments(segments: List[Segment]) -> Segment:
    """Combine multiple segments into a single segment."""
    if not segments:
        raise ValueError("Cannot combine empty segment list")

    if len(segments) == 1:
        return segments[0]

    # Combine texts with spaces
    combined_text = " ".join(seg.text for seg in segments)

    # Use earliest start and latest end
    earliest_start = min(seg.start for seg in segments)
    latest_end = max(seg.end for seg in segments)

    # Use the first segment's metadata but update timing and text
    first_seg = segments[0]
    return Segment(
        episode_id=first_seg.episode_id,
        start=earliest_start,
        end=latest_end,
        speaker=first_seg.speaker,
        text=combined_text,
        order=first_seg.order,
        source_path=first_seg.source_path
    )

def match_answer_to_source(answer: str, speaker_segments: Iterable[Segment]) -> Optional[str]:
    segments = list(speaker_segments)
    if not segments:
        return None

    merged_segments = merge_contiguous_segments(segments, max_gap=1.0, max_length=1200)
    if not merged_segments:
        return None

    combined_text = " ".join(seg.text.strip() for seg in merged_segments if seg.text).strip()
    if not combined_text:
        return None

    match = find_verbatim_span(combined_text, answer)
    if match is not None:
        matched_slice, _, _ = match
        return matched_slice

    if answer_present(combined_text, answer):
        fallback = answer.strip()
        return fallback if fallback else combined_text

    return None

# HARD CONSTRAINTS (do not change unless failing tests):
# - Allowed answer speakers: ["Fr Stephen De Young","Jonathan Pageau"] only.
# - Answers prefer verbatim substrings; near-matches pass via normalized similarity fallback.
# - Merge adjacent turns only if gap ≤ 1.0s; cap answer length ≤ 1200 chars.
# - Question source: prefer nearby non-target context; call gpt-4o-mini "questionify" ONLY if no clean question exists. Enforce ≤25 words and trailing "?".
# - Harmony-STRICT training output = JSONL with ONLY `messages` (system optional), last role MUST be "assistant".
# - Sidecar metadata JSONL is separate and 1:1 aligned with training lines. Never include sidecar fields in training JSONL.
# - Maintain current train/val split UNCHANGED unless checks fail:
#   • determinism with fixed seed,
#   • no overlap of pair_id/dedup_key between splits,
#   • (prefer) episode-level isolation.
# - Minimize model calls; local heuristics first; gpt-4o-mini only behind gates.

QUESTION_STARTERS = (
    "who",
    "what",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "do",
    "does",
    "did",
    "can",
    "could",
    "should",
    "would",
    "will",
    "may",
    "might",
    "shall",
    "have",
    "has",
    "had",
)

QUESTION_ALLOWED_EXTRAS: Set[str] = {
    "a",
    "about",
    "all",
    "and",
    "any",
    "are",
    "be",
    "can",
    "could",
    "do",
    "does",
    "did",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "let",
    "lets",
    "may",
    "might",
    "of",
    "on",
    "should",
    "shall",
    "that",
    "the",
    "their",
    "them",
    "these",
    "they",
    "this",
    "those",
    "to",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}

_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")
QUESTION_NEW_WORD_BUDGET = 4


def _is_question(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith("?"):
        return True
    match = _WORD_PATTERN.search(stripped.lower())
    if not match:
        return False
    first_word = match.group(0).lstrip("'")
    first_word = first_word.replace("'", "")
    return first_word in QUESTION_STARTERS


def _word_set(text: str) -> Set[str]:
    words: Set[str] = set()
    for token in _WORD_PATTERN.findall(text.lower()):
        cleaned = token.strip("'").replace("'", "")
        if cleaned:
            words.add(cleaned)
    return words


CONNECTOR_VERBS: Tuple[str, ...] = ("signals", "expresses", "reveals", "echoes", "maps to")
CONNECTOR_MOTIFS: Tuple[str, ...] = (
    "order vs chaos",
    "sacrifice and renewal",
    "fall and redemption",
    "exile and return",
)


def _deterministic_int(seed: str) -> int:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _select_variant_mode(pair_id: str, config: CompositionConfig) -> str:
    ratio = max(0.0, min(1.0, config.variant_b_ratio))
    if ratio <= 0.0:
        return "A"
    if ratio >= 1.0:
        return "B"
    bucket = _deterministic_int(f"{pair_id}:{config.variant_seed}") % 100
    threshold = int(ratio * 100)
    return "B" if bucket < threshold else "A"


def _build_connector_sentences(
    answer_text: str,
    speaker: str,
    config: CompositionConfig,
    seed_material: str,
) -> List[str]:
    if config.connector_max_sentences <= 0 or config.paraphrase_char_cap <= 0:
        return []

    base_seed = _deterministic_int(seed_material)
    max_sentences = max(0, config.connector_max_sentences)
    requested = min(max_sentences, 1 + (base_seed % max(1, max_sentences)))

    subject = speaker.strip() if speaker.strip() else "These lines"
    sentences: List[str] = []
    for idx in range(requested):
        offset = base_seed + idx
        verb = CONNECTOR_VERBS[offset % len(CONNECTOR_VERBS)]
        motif = CONNECTOR_MOTIFS[offset % len(CONNECTOR_MOTIFS)]
        if idx == requested - 1 and requested > 1:
            sentence = f"It leaves a question about {motif}."
        else:
            sentence = f"{subject} {verb} {motif} within the cited lines."
        sentences.append(sentence)

    trimmed: List[str] = []
    total_chars = 0
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        additional = len(sentence) if not trimmed else len(sentence) + 1
        if total_chars + additional > config.paraphrase_char_cap:
            break
        trimmed.append(sentence)
        total_chars += additional

    return trimmed


def _validate_question(question: str, context_text: Optional[str], answer_text: str) -> bool:
    stripped = question.strip()
    if not stripped or not stripped.endswith("?"):
        return False

    # Simple word-count check: ignore trailing question mark
    word_count = len([part for part in stripped[:-1].split() if part])
    if word_count > 25:
        return False

    if context_text is None:
        return True

    allowed_words = _word_set(context_text + " " + answer_text)
    allowed_words.update(QUESTION_ALLOWED_EXTRAS)

    new_word_budget = QUESTION_NEW_WORD_BUDGET

    for word in _word_set(stripped):
        if word.isdigit():
            continue
        if word in allowed_words:
            continue
        if word.rstrip("s") in allowed_words:
            continue
        if new_word_budget > 0:
            new_word_budget -= 1
            continue
        return False
    return True


def extract_question_context(window: ContextWindow) -> Optional[Tuple[str, str]]:
    """Return the contiguous pre-anchor utterance from a non-target speaker."""
    anchor_speaker = window.segments[window.anchor_index].speaker
    idx = window.anchor_index - 1
    collected: List[str] = []
    context_speaker: Optional[str] = None

    while idx >= 0:
        seg = window.segments[idx]
        if not seg.text.strip():
            idx -= 1
            continue

        if seg.speaker.lower() == anchor_speaker.lower():
            if collected:
                break
            idx -= 1
            continue

        if context_speaker is None:
            context_speaker = seg.speaker
        if seg.speaker.lower() != context_speaker.lower():
            break

        collected.append(seg.text.strip())
        if len(" ".join(reversed(collected)).split()) >= 60:
            break
        idx -= 1

    if not collected or context_speaker is None:
        return None

    collected.reverse()
    context_text = " ".join(collected).strip()
    if not context_text:
        return None
    return context_text, context_speaker


def _passes_questionify_gate(text: str) -> bool:
    """Return True if the context should be sent to questionify."""
    stripped = (text or "").strip()
    if not stripped:
        return False
    if "?" in stripped:
        return False
    length = len(stripped)
    if length < QUESTIONIFY_MIN_LENGTH or length > QUESTIONIFY_MAX_LENGTH:
        return False
    normalized = stripped.lstrip(' "\'“”‘’¿¡([{-').lower()
    for prefix in QUESTIONIFY_BLOCKED_PREFIXES:
        if normalized.startswith(prefix + " "):
            return False
        if normalized == prefix:
            return False
    return True


def questionify_context(
    client: OpenAI,
    config: QuestionifyConfig,
    context_text: str,
    answer_text: str,
) -> Optional[str]:
    if not config.enabled or not config.model:
        return None
    original_client = llm_questionify_client.client
    swapped = False
    try:
        if client is not None and client is not original_client:
            llm_questionify_client.client = client
            swapped = True
        result = llm_questionify_client.questionify(answer_text, context_text)
    except Exception:
        return None
    finally:
        if swapped:
            llm_questionify_client.client = original_client

    if not result:
        return None
    question = (result.get("question") or "").strip()
    return question or None


def _build_questionify_cache_key(window: ContextWindow, window_index: int, context_text: str) -> Tuple[str, int, str]:
    base = context_text.strip()
    context_hash = hashlib.sha256(base.encode("utf-8")).hexdigest()[:16] if base else ""
    return (window.episode_id, window_index, context_hash)


def _questionify_cache_get(
    cache: "OrderedDict[Tuple[str, int, str], str]",
    key: Optional[Tuple[str, int, str]],
) -> Optional[str]:
    if key is None:
        return None
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _questionify_cache_set(
    cache: "OrderedDict[Tuple[str, int, str], str]",
    key: Optional[Tuple[str, int, str]],
    value: str,
    capacity: int,
) -> None:
    if key is None or not value or capacity <= 0:
        return
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > capacity:
        cache.popitem(last=False)


def refine_question(
    original_question: str,
    window: ContextWindow,
    answer_text: str,
    answer_speaker: str,
    config: QuestionifyConfig,
    questionify_fn: Callable[[ContextWindow, str, str], Optional[str]],
    local_question: Optional[str],
    local_question_source: Optional[str],
    local_question_speaker: Optional[str],
    questionify_cache: "OrderedDict[Tuple[str, int, str], str]",
    window_index: int,
    metrics: Dict[str, Any],
    cache_capacity: int,
    questionify_guard: bool,
    questionify_min_chars: int,
) -> Optional[Tuple[str, Optional[str], Optional[str], str, bool]]:
    """Derive a compact interrogative that aligns with the answer and context."""
    model_candidate = original_question.strip()
    context_info = extract_question_context(window)
    context_text: Optional[str] = None
    context_speaker: Optional[str] = None

    if context_info:
        context_text, context_speaker = context_info

    if local_question:
        local_candidate = local_question.strip()
        if local_candidate:
            if not local_candidate.endswith("?"):
                local_candidate = f"{local_candidate}?"
            if _validate_question(local_candidate, local_question_source, answer_text):
                return local_candidate, local_question_source, local_question_speaker, "local", False

    questionify_context_text = context_text or local_question_source or ""
    questionify_context_speaker = context_speaker or local_question_speaker

    if context_text and _is_question(context_text):
        normalized_context = context_text if context_text.endswith("?") else f"{context_text}?"
        if _validate_question(normalized_context, context_text, answer_text):
            return normalized_context, context_text, context_speaker, "context", False

    if model_candidate and _is_question(model_candidate):
        normalized_model = model_candidate if model_candidate.endswith("?") else f"{model_candidate}?"
        if _validate_question(normalized_model, context_text, answer_text):
            return normalized_model, context_text, context_speaker, "model", False

    usable_context = (questionify_context_text or "").strip()
    if questionify_guard and len(usable_context) < questionify_min_chars:
        usable_context = ""

    should_questionify = (
        local_question is None
        and config.enabled
        and config.model
        and usable_context
        and _passes_questionify_gate(usable_context)
    )

    if should_questionify:
        cache_key = _build_questionify_cache_key(window, window_index, usable_context) if cache_capacity else None
        cached = _questionify_cache_get(questionify_cache, cache_key) if cache_capacity else None
        if cached:
            metrics["openai_cache_hits"] += 1
            if _validate_question(cached, usable_context, answer_text):
                return cached, usable_context, questionify_context_speaker, "model", True
            if cache_key is not None and cache_capacity:
                questionify_cache.pop(cache_key, None)
        metrics["openai_calls"] += 1
        try:
            rewritten = questionify_fn(window, usable_context, answer_text)
        except Exception:
            return None
        if rewritten:
            if _validate_question(rewritten, usable_context, answer_text):
                _questionify_cache_set(questionify_cache, cache_key, rewritten, cache_capacity)
                return rewritten, usable_context, questionify_context_speaker, "model", True
            if cache_key is not None and cache_capacity:
                questionify_cache.pop(cache_key, None)

    if model_candidate:
        normalized_model = model_candidate if model_candidate.endswith("?") else f"{model_candidate}?"
        if _validate_question(normalized_model, context_text, answer_text):
            return normalized_model, context_text, context_speaker, "model", False

    return None


def collect_source_span_texts(
    window: ContextWindow,
    speaker: str,
    citations: Sequence[Dict[str, Any]],
    fallback_text: str,
) -> List[str]:
    """Return ordered transcript spans for provenance checks."""
    spans: List[str] = []
    normalized_speaker = speaker.lower()

    for citation in citations:
        start = citation.get("start")
        end = citation.get("end")
        if start is None or end is None:
            continue

        try:
            start_val = float(start)
            end_val = float(end)
        except (TypeError, ValueError):
            continue

        if math.isnan(start_val) or math.isnan(end_val):
            continue

        collected: List[str] = []
        for seg in window.segments:
            if seg.speaker.lower() != normalized_speaker:
                continue
            if math.isnan(seg.start) or math.isnan(seg.end):
                continue
            if seg.end < start_val or seg.start > end_val:
                continue
            text = seg.text.strip()
            if text:
                collected.append(text)

        if collected:
            spans.append(" ".join(collected).strip())

    if spans:
        return [span for span in spans if span]

    fallback = fallback_text.strip()
    return [fallback] if fallback else []


def collect_source_spans(
    window: ContextWindow,
    speaker: str,
    citations: Sequence[Dict[str, Any]],
    fallback_text: str,
) -> str:
    """Build a transcript excerpt to supply as SOURCE_SPANS for validation."""
    span_texts = collect_source_span_texts(window, speaker, citations, fallback_text)
    return "\n".join(span_texts)


def provenance_norm(text: str) -> str:
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2192": "->",
    }
    normalized = text.lower()
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def validate_provenance(
    span_texts: Sequence[str],
    answer_text: str,
    connectors_text: str = "",
) -> bool:
    span_norm = " ".join(provenance_norm(span) for span in span_texts if provenance_norm(span))
    answer_norm = provenance_norm(answer_text)
    if connectors_text:
        connectors_norm = provenance_norm(connectors_text)
        expected = f"{span_norm} {connectors_norm}".strip()
    else:
        expected = span_norm.strip()
    return answer_norm == expected


def pairfit_prompt(question: str, answer: str, source_spans: str) -> str:
    return (
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER (verbatim from transcript):\n"
        f"{answer}\n\n"
        "SOURCE_SPANS (excerpted transcript backing the answer):\n"
        f"{source_spans}\n\n"
        "Return:\n"
        "{\n"
        '  "is_supported": true|false,\n'
        '  "is_good_question": true|false,\n'
        '  "reason": "one short sentence",\n'
        '  "suggested_question": "<rewrite <=25 words or empty string>"\n'
        "}"
    )


def call_pairfit_judge(
    client: OpenAI,
    config: PairFitConfig,
    question: str,
    answer: str,
    source_spans: str,
) -> Optional[Dict[str, Any]]:
    if not config.enabled or not config.model:
        return None

    system_prompt = (
        "You judge whether ANSWER is directly supported by SOURCE_SPANS, and whether QUESTION is a natural "
        "question for that answer.\nBe strict. Use only evidence in SOURCE_SPANS/ANSWER text.\nReturn JSON only."
    )

    response = client.responses.create(
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": pairfit_prompt(question, answer, source_spans)},
        ],
    )

    payload = _extract_json_payload(response.output_text)  # type: ignore[attr-defined]
    if not payload:
        return None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def get_cross_encoder():
    """Expose the shared cross-encoder instance for tests and diagnostics."""
    return _pairfit_get_cross_encoder()


def compute_semantic_fit_score(question: str, answer: str, _context: Optional[str] = None) -> Optional[float]:
    """Compute semantic fit score using global cross-encoder helper."""
    scores = score_pairs_ce([(question, answer)])
    if not scores:
        return None
    return scores[0]


def evaluate_pairfit(
    question: str,
    answer: str,
    source_spans: str,
    context_reference: Optional[str],
    config: PairFitConfig,
    pairfit_fn: Callable[[str, str, str], Optional[Dict[str, Any]]],
) -> Tuple[Optional[str], Dict[str, Any]]:
    """Return (question, metadata) if accepted; otherwise (None, metadata)."""
    if not config.enabled or not config.model:
        return question, {"status": "skipped"}

    attempts: List[Dict[str, Any]] = []
    candidate = question.strip()

    for attempt in range(2):
        ce_score_raw = compute_semantic_fit_score(candidate, answer)
        if ce_score_raw is None:
            attempts.append({"question": candidate, "error": "ce_unavailable"})
            return None, {
                "status": "rejected",
                "attempts": attempts,
                "reason": "Cross-encoder unavailable - cannot validate semantic fit.",
            }
        fit_score_ce = round(ce_score_raw, 3)

        result = pairfit_fn(candidate, answer, source_spans)
        metadata_attempt: Dict[str, Any] = {"question": candidate, "fit_score_ce": fit_score_ce}

        if not isinstance(result, dict):
            metadata_attempt.update({"error": "invalid_response"})
            attempts.append(metadata_attempt)
            return None, {
                "status": "error",
                "attempts": attempts,
                "reason": "PairFit judge returned invalid JSON.",
                "fit_score_ce": fit_score_ce,
            }

        supported = bool(result.get("is_supported"))
        good_question = bool(result.get("is_good_question"))
        reason = str(result.get("reason") or "").strip()
        suggested = str(result.get("suggested_question") or "").strip()

        metadata_attempt.update(
            {
                "is_supported": supported,
                "is_good_question": good_question,
                "reason": reason or None,
                "suggested_question": suggested or None,
            }
        )
        attempts.append(metadata_attempt)

        # Phase 2: Enhanced pairfit logic with cross-encoder scores
        pairfit_passed = supported and good_question and fit_score_ce >= 0.5
        if pairfit_passed:
            return candidate, {
                "status": "accepted",
                "attempts": attempts,
                "fit_score_ce": fit_score_ce,
            }

        if attempt == 0 and suggested:
            suggestion = suggested if suggested.endswith("?") else f"{suggested.strip()}?"
            suggestion = suggestion.strip()
            reference = context_reference or source_spans
            if not _validate_question(suggestion, reference, answer):
                return None, {
                    "status": "rejected",
                    "attempts": attempts,
                    "reason": "Suggested question failed validation.",
                    "fit_score_ce": fit_score_ce,
                }

            should_retry = False
            if (supported and good_question and 0.45 <= fit_score_ce < 0.5) or not (supported and good_question):
                should_retry = True

            if should_retry:
                candidate = suggestion
                continue

        return None, {
            "status": "rejected",
            "attempts": attempts,
            "reason": reason or "PairFit judge rejected pair.",
            "fit_score_ce": fit_score_ce,
        }

    return None, {
        "status": "rejected",
        "attempts": attempts,
        "reason": "PairFit judge did not accept pair after retry.",
        "fit_score_ce": fit_score_ce,
    }


def count_words(text: str) -> int:
    return len([token for token in text.strip().split() if token])


def should_trim_answer(answer: str, threshold: int) -> bool:
    return count_words(answer) > threshold


def call_answer_trimmer(
    client: OpenAI,
    config: AnswerTrimConfig,
    question: str,
    answer: str,
) -> Optional[str]:
    if not config.enabled or not config.model:
        return None

    system_prompt = (
        "Trim ANSWER to the shortest excerpt that still fully answers QUESTION.\n"
        "Do not add words not present in the answer. Delete filler only.\n"
        "<= 150 tokens.\n"
        "Return JSON only."
    )

    user_prompt = (
        "QUESTION:\n"
        f"{question}\n\n"
        "ANSWER (verbatim):\n"
        f"{answer}\n\n"
        'Return:\n{"answer_trimmed": "<subset-of-answer-text>"}'
    )

    response = client.responses.create(
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    payload = _extract_json_payload(response.output_text)  # type: ignore[attr-defined]
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    trimmed = str(data.get("answer_trimmed") or "").strip()
    if not trimmed:
        return None
    return trimmed


def apply_answer_trim(
    question: str,
    original_answer: str,
    config: AnswerTrimConfig,
    trim_fn: Callable[[str, str], Optional[str]],
) -> Optional[Dict[str, Any]]:
    if not config.enabled or not config.model:
        return None
    if not should_trim_answer(original_answer, config.threshold_tokens):
        return None

    try:
        trimmed = trim_fn(question, original_answer)
    except Exception:
        return None
    if not trimmed:
        return None

    candidate = trimmed.strip()
    if not candidate:
        return None
    start_idx = original_answer.find(candidate)
    if start_idx == -1:
        return None

    end_idx = start_idx + len(candidate)
    if len(candidate) >= len(original_answer):
        return None

    return {
        "text": candidate,
        "start_char": start_idx,
        "end_char": end_idx,
        "original_length": len(original_answer),
        "trimmed_length": len(candidate),
    }


def call_speaker_check(
    client: OpenAI,
    config: SpeakerCheckConfig,
    allowed_speakers: Sequence[str],
    answer_speaker: str,
    answer_text: str,
) -> Optional[Dict[str, Any]]:
    if not config.enabled or not config.model:
        return None

    system_prompt = (
        "Return JSON stating whether ANSWER_SPEAKER is allowed (one of the whitelisted names) "
        "and that ANSWER contains only their words. If not certain, return allowed=false."
    )
    allowed_list = "[" + ", ".join(f'"{name}"' for name in allowed_speakers) + "]"
    user_prompt = (
        f'ALLOWED_SPEAKERS: {allowed_list}\n'
        f'ANSWER_SPEAKER: "{answer_speaker}"\n'
        "ANSWER:\n"
        f"{answer_text}\n\n"
        'Return:\n{"allowed": true|false, "reason": "short"}'
    )

    response = client.responses.create(
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    payload = _extract_json_payload(response.output_text)  # type: ignore[attr-defined]
    if not payload:
        return None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None
    allowed = bool(data.get("allowed"))
    reason = str(data.get("reason") or "").strip()
    return {"allowed": allowed, "reason": reason or None}


def call_qc_audit(
    client: OpenAI,
    config: QCAuditConfig,
    record_payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not config.enabled or not config.model:
        return None

    system_prompt = "You audit dataset items for finetuning. Be strict and concise.\nReturn JSON only."
    user_prompt = (
        "Record:\n"
        f"{json.dumps(record_payload, ensure_ascii=False, indent=2)}\n\n"
        "Return:\n"
        "{\n"
        '  "pass": true|false,\n'
        '  "issues": ["unsupported","not_a_question","too_long_answer","speaker_not_allowed","duplicate_like","other"],\n'
        '  "comment": "=120 chars"\n'
        "}"
    )

    response = client.responses.create(
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    payload = _extract_json_payload(response.output_text)  # type: ignore[attr-defined]
    if not payload:
        return None

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    passed = bool(data.get("pass"))
    issues = data.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    issues = [str(issue) for issue in issues][:6]
    comment = str(data.get("comment") or "").strip()

    return {"pass": passed, "issues": issues, "comment": comment}


def sample_qc_indices(
    total_records: int,
    sample_rate: float,
    min_samples: int,
    seed: int,
) -> List[int]:
    if total_records <= 0 or sample_rate <= 0:
        return []
    sample_rate = max(0.0, min(sample_rate, 1.0))
    count = max(min_samples if total_records > 0 else 0, int(round(total_records * sample_rate)))
    count = min(count, total_records)
    if count <= 0:
        return []
    rng = random.Random(seed)
    indices = list(range(total_records))
    rng.shuffle(indices)
    return sorted(indices[:count])


def run_qc_audit(
    client: OpenAI,
    config: QCAuditConfig,
    records: List[Dict[str, Any]],
    audit_rows: List[Dict[str, Any]],
) -> None:
    if not config.enabled or not config.model:
        return
    total = len(records)
    if total == 0:
        return

    indices = sample_qc_indices(total, config.sample_rate, config.min_samples, config.seed)
    if not indices:
        return

    failures: List[Dict[str, Any]] = []
    passed = 0
    evaluated = 0

    for idx in indices:
        record = records[idx]
        metadata = record.get("metadata", {})
        question = metadata.get("question") or ""
        answer = metadata.get("answer") or ""
        answer_speaker = metadata.get("answer_speaker") or ""
        episode_id = metadata.get("episode_id") or ""
        source_spans = metadata.get("anchor_segments") or []

        record_payload = {
            "episode_id": episode_id,
            "question": question,
            "answer": answer,
            "answer_speaker": answer_speaker,
            "source_spans": source_spans,
        }

        qc_result = call_qc_audit(client, config, record_payload)
        if not isinstance(qc_result, dict):
            qc_result = {"pass": None, "issues": ["parse_failure"], "comment": "qc parse failure"}
            metadata["qc_audit"] = qc_result
            if idx < len(audit_rows):
                audit_rows[idx]["qc_status"] = "error"
            first_source = (metadata.get("sources") or [{}])[0]
            timestamp = format_timestamp(first_source.get("start", math.nan))
            failures.append(
                {
                    "index": idx,
                    "episode_id": episode_id,
                    "window_id": metadata.get("window_id"),
                    "answer_speaker": answer_speaker,
                    "timestamp": timestamp,
                    "issues": qc_result.get("issues"),
                    "comment": qc_result.get("comment"),
                }
            )
            continue

        metadata["qc_audit"] = qc_result
        audit_status = "pass" if qc_result.get("pass") else "fail"
        if idx < len(audit_rows):
            audit_rows[idx]["qc_status"] = audit_status

        evaluated += 1

        if qc_result.get("pass"):
            passed += 1
        else:
            first_source = (metadata.get("sources") or [{}])[0]
            timestamp = format_timestamp(first_source.get("start", math.nan))
            failures.append(
                {
                    "index": idx,
                    "episode_id": episode_id,
                    "window_id": metadata.get("window_id"),
                    "answer_speaker": answer_speaker,
                    "timestamp": timestamp,
                    "issues": qc_result.get("issues"),
                    "comment": qc_result.get("comment"),
                }
            )

    sample_count = evaluated
    pass_rate = passed / sample_count if sample_count else 1.0

    if config.output_path:
        output_path = config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sample_size": sample_count,
                    "pass_rate": pass_rate,
                    "threshold": config.pass_threshold,
                    "failures": failures,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    if pass_rate < config.pass_threshold:
        raise RuntimeError(
            f"QC audit failed: pass rate {pass_rate:.2%} below threshold {config.pass_threshold:.2%}"
        )


def build_prompt(window: ContextWindow, whitelist: Sequence[str], questions_per_window: int) -> str:
    context_lines = [format_segment(seg) for seg in window.segments]
    anchor_seg = window.segments[window.anchor_index]
    anchor_info = (
        f"Anchor speaker: {anchor_seg.speaker} "
        f"({format_timestamp(anchor_seg.start)}-{format_timestamp(anchor_seg.end)})"
    )
    context_speakers = sorted({seg.speaker for seg in window.segments})
    other_speakers = [s for s in context_speakers if s.lower() != anchor_seg.speaker.lower()]
    other_clause = ""
    if other_speakers:
        other_clause = (
            " Do not borrow wording from these other voices: "
            + ", ".join(other_speakers)
            + "."
        )
    return (
        "You are preparing faithful question/answer pairs for fine-tuning an LLM.\n"
        "Rules:\n"
        "- Craft one question that a thoughtful listener might ask after hearing the highlighted teaching.\n"
        "  * Focus on the theological or pastoral theme in the excerpt.\n"
        "  * Do NOT mention the speaker's name in the question; ask generically (e.g., \"How should we...?\").\n"
        "- Answers must consist solely of the anchor speaker's sentences (verbatim or lightly trimmed)."
        + other_clause
        + "\n"
        "  * Copy contiguous sentence(s) exactly as they appear; do not paraphrase or omit interior phrases.\n"
        "  * If you need to trim filler (\"uh\", \"you know\"), only remove those words and keep the remaining text intact.\n"
        "- Whitelist of answer speakers: "
        + ", ".join(whitelist)
        + "\n"
        f"- Approved voice for the answer: {anchor_seg.speaker}. Do not add framing like \"{anchor_seg.speaker} says\".\n"
        "- Provide a brief analysis explaining why the quoted answer resolves the question and cite the supporting lines.\n"
        '- Return JSON with structure:\n{"pairs":[{"question":"...","answer":"...","answer_speaker":"...","analysis":"...",'
        '"citations":[{"speaker":"...","start":number,"end":number}]}]}\n'
        "- Produce up to "
        f"{questions_per_window} "
        "pairs. If no faithful pair is possible, return {\"pairs\":[]}.\n\n"
        f"{anchor_info}\n\nTranscript excerpt:\n"
        + "\n".join(context_lines)
    )


def _extract_json_payload(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    if fence_match:
        return fence_match.group(1).strip()

    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) > 1:
            body = "\n".join(lines[1:]).strip()
            if body.endswith("```"):
                body = "\n".join(body.splitlines()[:-1]).strip()
            return body
    return stripped


def _normalise_numbers(payload: str) -> str:
    def convert_timestamp(match: re.Match[str]) -> str:
        minutes = int(match.group(1))
        seconds = float(match.group(2))
        total = minutes * 60 + seconds
        return f'": {total}'

    payload = re.sub(r'":\s*(\d{1,3}):(\d{2}(?:\.\d+)?)', convert_timestamp, payload)
    payload = re.sub(r":\s*0([1-9]\d*(?:\.\d+)?)", r": \1", payload)
    return payload


def call_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    max_output_tokens: int,
    temperature: float,
) -> Dict:
    response = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input=[
            {"role": "system", "content": "You generate faithful question/answer data for Harmony fine-tuning."},
            {"role": "user", "content": prompt},
        ],
    )
    text = response.output_text  # type: ignore[attr-defined]
    payload = _extract_json_payload(text)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        fixed_payload = _normalise_numbers(payload)
        if fixed_payload != payload:
            try:
                return json.loads(fixed_payload)
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Failed to parse model output as JSON: {exc}\n\n{text}") from exc


def build_harmony_record(
    pair: Dict,
    window: ContextWindow,
    sample_idx: int,
    composition_config: CompositionConfig,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Build Harmony-STRICT training record and separate sidecar metadata."""
    pair_id = f"{window.episode_id}_{window.window_id}_{sample_idx}"
    answer_speaker = pair.get("answer_speaker", "Unknown")
    anchor_segments = [
        seg for seg in window.segments if seg.speaker.lower() == answer_speaker.lower()
    ]
    if not anchor_segments:
        anchor_segments = list(window.segments)
    context_excerpt = "\n".join(seg.text for seg in anchor_segments)
    question = pair["question"].strip()
    answer = pair["answer"].strip()
    span_texts = [
        text.strip()
        for text in pair.get("_answer_spans_text", [])
        if isinstance(text, str) and text.strip()
    ]
    base_answer = " ".join(span_texts).strip() if span_texts else answer
    if not base_answer:
        raise ValueError(f"Empty base answer for {pair_id}")
    if not span_texts:
        span_texts = [base_answer]
    answer = base_answer
    analysis = pair.get("analysis", "").strip() or "Answer quotes the cited speaker verbatim with optional connective phrasing."
    sources = pair.get("citations", [])
    question_origin = pair.get("_question_origin", "model")
    question_context = pair.get("_question_context")
    question_context_speaker = pair.get("_question_context_speaker")

    variant_mode = _select_variant_mode(pair_id, composition_config)
    connector_sentences = 0
    paraphrase_chars = 0
    connectors_text = ""
    connectors: List[str] = []
    if variant_mode == "B":
        seed_material = f"{pair_id}:{composition_config.variant_seed}:{len(answer)}"
        connectors = _build_connector_sentences(
            answer_text=answer,
            speaker=answer_speaker,
            config=composition_config,
            seed_material=seed_material,
        )
        connector_sentences = len(connectors)
        connectors_text = " ".join(connectors).strip()
        paraphrase_chars = len(connectors_text)
        if connector_sentences == 0 or paraphrase_chars == 0:
            variant_mode = "A"
            connector_sentences = 0
            paraphrase_chars = 0
            connectors_text = ""
            assistant_answer = answer
        else:
            candidate_answer = f"{answer}\n\n{connectors_text}"
            if validate_provenance(span_texts, candidate_answer, connectors_text):
                assistant_answer = candidate_answer
            else:
                variant_mode = "A"
                connector_sentences = 0
                paraphrase_chars = 0
                connectors_text = ""
                connectors = []
                assistant_answer = answer
    else:
        assistant_answer = answer

    if variant_mode == "A" and not validate_provenance(span_texts, assistant_answer):
        raise ValueError(f"Provenance mismatch for {pair_id} (Variant A).")
    if variant_mode == "B" and connectors_text and not validate_provenance(span_texts, assistant_answer, connectors_text):
        raise ValueError(f"Provenance mismatch for {pair_id} (Variant B).")

    # Build system message (optional in Harmony-STRICT)
    system_message = (
        "You are a symbolic commentator in the theological voice of the Lord of Spirits / Symbolic World corpus. "
        "Ground everything in the provided transcript excerpts from Fr Stephen De Young or Jonathan Pageau. "
        "Prefer quotes and light connective phrases; no invention. Identify patterns (order/chaos, sacrifice/renewal, fall/redemption) "
        "and conclude with a compact paradox, lesson, or question."
    )
    user_message = (
        f"Question: {question}\n\nContext excerpt:\n{context_excerpt}\n"
        "Base your answer solely on the cited transcript text."
    )
    assistant_analysis = (
        f"{analysis}\nSources: "
        + ", ".join(
            f"{src.get('speaker', '-')}@{format_timestamp(src.get('start', math.nan))}"
            for src in sources
        )
    ).strip()

    # HARMONY-STRICT TRAINING RECORD: ONLY messages, last role MUST be assistant
    training_record = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": f"{assistant_analysis}\n\n{assistant_answer}"},
        ]
    }

    pair_validation = pair.get("_pair_validation") or {}
    fit_score_ce_value: Optional[float] = None
    if pair_validation.get("fit_score_ce") is not None:
        fit_score_ce_value = round(float(pair_validation["fit_score_ce"]), 3)

    anchor_source_path: Optional[Path] = None
    if window.segments:
        anchor_idx = min(max(window.anchor_index, 0), len(window.segments) - 1)
        anchor_source_path = window.segments[anchor_idx].source_path
        if anchor_source_path is None:
            for seg in window.segments:
                if getattr(seg, "source_path", None) is not None:
                    anchor_source_path = seg.source_path
                    break

    if anchor_source_path is None:
        raise ValueError("Missing source_file for sidecar record.")

    # SIDECAR METADATA: Separate JSONL with 1:1 alignment
    quoted_spans: List[Dict[str, Any]] = []
    source_iter = sources if isinstance(sources, list) else []
    for src in source_iter:
        if not isinstance(src, dict):
            continue
        span_speaker = src.get("speaker", "")
        span_start = src.get("start")
        span_end = src.get("end")
        quoted_spans.append(
            {
                "speaker": span_speaker,
                "start": float(span_start) if isinstance(span_start, (int, float)) else None,
                "end": float(span_end) if isinstance(span_end, (int, float)) else None,
            }
        )

    if variant_mode != "B":
        connector_sentences = 0
        paraphrase_chars = 0

    composition_record = {
        "mode": variant_mode,
        "connector_sentences": connector_sentences,
        "paraphrase_chars": paraphrase_chars,
        "quoted_spans": quoted_spans,
        "answer_spans_text": list(span_texts),
    }

    sidecar_metadata = {
        "pair_id": pair_id,
        "episode_id": window.episode_id,
        "window_id": window.window_id,
        "sample_index": sample_idx,
        "answer_speaker": answer_speaker,
        "question": question,
        "answer": answer,
        "sources": sources,
        "answer_found_in_source": bool(pair.get("_answer_found_in_source", False)),
        "answer_spans_text": list(span_texts),
        "anchor_segments": [
            {
                "speaker": seg.speaker,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in anchor_segments
        ],
        "origin_segments": [
            {
                "speaker": seg.speaker,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            }
            for seg in window.segments
        ],
        "question_origin": question_origin,
        "target_spans": sources,  # For validation
        "source_file": str(anchor_source_path) if anchor_source_path else None,
        "fit_score_ce": fit_score_ce_value,
        "gates": {
            "questionify_used": bool(pair.get("_questionify_used", False)),
            "answer_trimmed": pair.get("_answer_trim") is not None,
            "speaker_check_passed": pair.get("_speaker_check", {}).get("allowed", False) if pair.get("_speaker_check") else False,
            "pairfit_passed": bool(fit_score_ce_value is not None and fit_score_ce_value >= 0.5),
        },
        "composition": composition_record,
    }

    # VALIDATE: Gate consistency enforcement
    pairfit_passed = sidecar_metadata["gates"]["pairfit_passed"]
    fit_score_ce = sidecar_metadata["fit_score_ce"]

    if pairfit_passed and fit_score_ce is not None and fit_score_ce < 0.5:
        raise ValueError(f"Gate consistency violation: pairfit_passed=True but fit_score_ce={fit_score_ce} < 0.5")
    if not pairfit_passed and fit_score_ce is not None and fit_score_ce >= 0.5:
        raise ValueError(f"Gate consistency violation: pairfit_passed=False but fit_score_ce={fit_score_ce} >= 0.5")

    # Add optional metadata fields
    if question_context:
        sidecar_metadata["question_context"] = question_context
    if question_context_speaker:
        sidecar_metadata["question_context_speaker"] = question_context_speaker
    if pair.get("_pair_validation"):
        sidecar_metadata["pair_validation"] = pair["_pair_validation"]
    if pair.get("_answer_trim"):
        sidecar_metadata["answer_trim"] = pair["_answer_trim"]
    if pair.get("_speaker_check"):
        sidecar_metadata["speaker_check"] = pair["_speaker_check"]
    if pair.get("_source_spans"):
        sidecar_metadata["source_spans_text"] = pair["_source_spans"]

    return training_record, sidecar_metadata


def split_records(records: List[Dict], train_split: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    rng.shuffle(records)
    cutoff = int(len(records) * train_split)
    if cutoff == len(records) and records:
        cutoff = len(records) - 1
    return records[:cutoff], records[cutoff:]


def write_jsonl_batch(path: Path, rows: Sequence[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for idx, record in enumerate(rows, 1):
            if orjson:
                payload = orjson.dumps(record)  # type: ignore[arg-type]
            else:
                payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
            handle.write(payload)
            handle.write(b"\n")
            if idx % 200 == 0:
                handle.flush()


def write_jsonl(path: Path, records: Sequence[Dict]) -> None:
    write_jsonl_batch(path, records)


def write_audit(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Harmony Q/A Generation Audit\n\n")
        handle.write("| Episode | Window | Question | Answer Speaker | Source Timestamp | Question Origin | PairFit | Speaker Check | QC |\n")
        handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
        for row in rows:
            pairfit_status = row.get("pairfit_status") or "-"
            question_origin = row.get("question_origin") or "-"
            speaker_status = row.get("speaker_check") or "-"
            qc_status = row.get("qc_status") or "-"
            handle.write(
                f"| {row['episode']} | `{row['window']}` | {row['question']} | "
                f"{row['answer_speaker']} | {row['source']} | {question_origin} | {pairfit_status} | {speaker_status} | {qc_status} |\n"
            )

def generate_records(args: argparse.Namespace) -> Tuple[Tuple[List[Dict], List[Dict]], Tuple[List[Dict], List[Dict]]]:
    """Generate Harmony-STRICT records and return (training_records, sidecar_metadata)."""

    prepare_cross_encoder(
        batch_size=getattr(args, "ce_batch_size", 256),
        max_length=getattr(args, "ce_max_length", 512),
    )

    segments = load_transcripts(args.input_glob)
    # HARD CONSTRAINT: Use exactly ["Fr Stephen De Young", "Jonathan Pageau"]
    ALLOWED_SPEAKERS = ["Fr Stephen De Young", "Jonathan Pageau"]
    answer_speakers = [speaker.strip() for speaker in args.answer_speakers.split(",") if speaker.strip()]

    # Override with correct allowed speakers
    if answer_speakers != ALLOWED_SPEAKERS:
        print(f"⚠️  Overriding answer_speakers with correct allowed speakers: {ALLOWED_SPEAKERS}")
        answer_speakers = ALLOWED_SPEAKERS

    if not answer_speakers:
        raise ValueError("At least one answer speaker must be provided.")

    windows = build_windows(
        segments=segments,
        answer_speakers=answer_speakers,
        context_turns=args.context_turns,
        max_context_tokens=args.max_context_tokens,
        window_stride=getattr(args, "window_stride", 1),
    )

    if args.dry_run_windows:
        windows = windows[: args.dry_run_windows]

    ensure_openai_api_key()
    client = OpenAI()
    llm_questionify_client.client = client
    questionify_model = (args.questionify_model or "").strip() if hasattr(args, "questionify_model") else ""
    questionify_enabled = bool(questionify_model) and not getattr(args, "disable_questionify", False)
    questionify_config = QuestionifyConfig(
        model=questionify_model if questionify_enabled else None,
        temperature=getattr(args, "questionify_temperature", 0.2),
        max_output_tokens=getattr(args, "questionify_max_output_tokens", 120),
        enabled=questionify_enabled,
    )
    questionify_guard = getattr(args, "questionify_guard", True)
    questionify_min_chars = max(0, int(getattr(args, "questionify_min_chars", 30)))

    def questionify_fn(window_obj: ContextWindow, context_text: str, answer_text: str) -> Optional[str]:
        if not questionify_enabled or not questionify_config.model:
            return None
        try:
            result = llm_questionify_client.questionify(answer_text, context_text)
        except Exception:
            return None
        if not result:
            return None
        question = (result.get("question") or "").strip()
        return question or None

    trim_model = (args.answer_trim_model or "").strip() if hasattr(args, "answer_trim_model") else ""
    trim_enabled = bool(trim_model) and not getattr(args, "disable_answer_trim", False)
    answer_trim_config = AnswerTrimConfig(
        model=trim_model if trim_enabled else None,
        temperature=getattr(args, "answer_trim_temperature", 0.1),
        max_output_tokens=getattr(args, "answer_trim_max_output_tokens", 160),
        threshold_tokens=getattr(args, "answer_trim_threshold", 150),
        enabled=trim_enabled,
    )

    def answer_trim_fn(question: str, answer_text: str) -> Optional[str]:
        return call_answer_trimmer(client, answer_trim_config, question, answer_text)

    speaker_model = (args.speaker_check_model or "").strip() if hasattr(args, "speaker_check_model") else ""
    speaker_enabled = bool(speaker_model) and not getattr(args, "disable_speaker_check", False)
    speaker_check_config = SpeakerCheckConfig(
        model=speaker_model if speaker_enabled else None,
        temperature=getattr(args, "speaker_check_temperature", 0.0),
        max_output_tokens=getattr(args, "speaker_check_max_output_tokens", 80),
        enabled=speaker_enabled,
    )

    def speaker_check_fn(answer_speaker: str, answer_text: str) -> Optional[Dict[str, Any]]:
        return call_speaker_check(
            client=client,
            config=speaker_check_config,
            allowed_speakers=answer_speakers,
            answer_speaker=answer_speaker,
            answer_text=answer_text,
        )

    qc_model = (args.qc_model or "").strip() if hasattr(args, "qc_model") else ""
    qc_enabled = bool(qc_model) and not getattr(args, "disable_qc_audit", False)
    qc_output_path = getattr(args, "qc_output_path", None)
    qc_path = qc_output_path if qc_output_path else None
    qc_config = QCAuditConfig(
        model=qc_model if qc_enabled else None,
        temperature=getattr(args, "qc_temperature", 0.0),
        max_output_tokens=getattr(args, "qc_max_output_tokens", 480),
        sample_rate=getattr(args, "qc_sample_rate", 0.02),
        min_samples=max(0, getattr(args, "qc_min_samples", 1)),
        pass_threshold=getattr(args, "qc_pass_threshold", 0.95),
        seed=getattr(args, "qc_seed", getattr(args, "seed", 7)),
        output_path=qc_path,
        enabled=qc_enabled,
    )

    pairfit_model = (args.pairfit_model or "").strip() if hasattr(args, "pairfit_model") else ""
    pairfit_enabled = bool(pairfit_model) and not getattr(args, "disable_pairfit", False)
    pairfit_config = PairFitConfig(
        model=pairfit_model if pairfit_enabled else None,
        temperature=getattr(args, "pairfit_temperature", 0.1),
        max_output_tokens=getattr(args, "pairfit_max_output_tokens", 180),
        enabled=pairfit_enabled,
    )

    def pairfit_fn(question: str, answer_text: str, source_spans: str) -> Optional[Dict[str, Any]]:
        return call_pairfit_judge(client, pairfit_config, question, answer_text, source_spans)

    composition_config = CompositionConfig(
        variant_b_ratio=max(0.0, min(1.0, float(getattr(args, "variant_b_ratio", 0.0)))),
        connector_max_sentences=max(0, int(getattr(args, "connector_max_sentences", 2))),
        paraphrase_char_cap=max(0, int(getattr(args, "paraphrase_char_cap", 300))),
        variant_seed=int(getattr(args, "variant_seed", 42)),
    )

    training_records: List[Dict] = []
    sidecar_metadata: List[Dict] = []
    audit_rows: List[Dict] = []
    total_pairs = 0

    # Phase 2: Logging and metrics tracking
    metrics = {
        "total_windows": len(windows),
        "total_pairs": 0,
        "kept_pairs": 0,
        "openai_calls": 0,
        "openai_cache_hits": 0,
        "windows_processed": 0,
        "preheuristic_questions": 0,
        "dropped_by_reason": {
            "allowlist_fail": 0,
            "fit_fail": 0,
            "trim_fail": 0,
            "speaker_check_fail": 0,
            "answer_not_found": 0,
            "question_refinement_fail": 0,
            "pair_validation_fail": 0,
            "provenance_fail": 0,
            "questionify_json_fail": 0,
        },
        "per_speaker_counts": {},
    }
    questionify_cache_capacity = max(0, int(getattr(args, "questionify_cache_size", QUESTIONIFY_CACHE_CAPACITY_DEFAULT)))
    if not questionify_enabled:
        questionify_cache_capacity = 0
    questionify_cache: OrderedDict[Tuple[str, int, str], str] = OrderedDict()

    start_time = time.time()
    last_log_time = start_time

    for window_idx, window in enumerate(windows, start=1):
        metrics["windows_processed"] += 1
        prompt = build_prompt(window, answer_speakers, args.questions_per_window)
        try:
            result = call_openai(
                client=client,
                model=args.model,
                prompt=prompt,
                max_output_tokens=args.max_output_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:  # pragma: no cover - network/parse failure
            print(f"[warn] Window {window.window_id}: generation failed ({exc})")
            continue

        pairs = result.get("pairs") or []
        if not isinstance(pairs, list):
            print(f"[warn] Window {window.window_id}: unexpected pairs payload")
            continue

        for pair in pairs:
            total_pairs += 1
            if not isinstance(pair, dict):
                continue
            speaker = pair.get("answer_speaker", "").strip()
            if speaker.lower() not in {s.lower() for s in answer_speakers}:
                metrics["dropped_by_reason"]["allowlist_fail"] += 1
                continue
            if not pair.get("question") or not pair.get("answer"):
                continue

            speaker_segments = [
                seg for seg in window.segments if seg.speaker.lower() == speaker.lower()
            ]
            matched_text = match_answer_to_source(pair["answer"], speaker_segments)
            if matched_text is None:
                print(
                    f"[warn] Window {window.window_id}: answer not found in source text; skipping.\n"
                    f"   answer: {pair['answer']!r}"
                )
                metrics["dropped_by_reason"]["answer_not_found"] += 1
                continue

            base_answer = matched_text.strip()
            span_texts = collect_source_span_texts(
                window=window,
                speaker=speaker,
                citations=pair.get("citations") or [],
                fallback_text=base_answer,
            )
            if not span_texts:
                print(
                    f"[warn] Window {window.window_id}: no transcript spans located for answer; skipping."
                )
                metrics["dropped_by_reason"]["answer_not_found"] += 1
                continue

            base_span_texts = list(span_texts)
            current_span_texts = list(base_span_texts)
            pair["answer"] = base_answer
            pair["_answer_spans_text"] = current_span_texts
            pair["_answer_found_in_source"] = True

            lookback_turns = min(3, max(1, args.context_turns))
            last_non_answer_text, last_non_answer_speaker = last_non_answer_snippet(
                window,
                lookback_turns,
                speaker,
            )
            local_question = mine_local_question(window, speaker, lookback_turns)

            prior_json_fail = metrics["dropped_by_reason"]["questionify_json_fail"]
            refinement = refine_question(
                original_question=pair.get("question", ""),
                window=window,
                answer_text=base_answer,
                answer_speaker=speaker,
                config=questionify_config,
                questionify_fn=questionify_fn,
                local_question=local_question,
                local_question_source=last_non_answer_text,
                local_question_speaker=last_non_answer_speaker,
                questionify_cache=questionify_cache,
                window_index=window_idx,
                metrics=metrics,
                cache_capacity=questionify_cache_capacity,
                questionify_guard=questionify_guard,
                questionify_min_chars=questionify_min_chars,
            )
            if refinement is None:
                if metrics["dropped_by_reason"]["questionify_json_fail"] > prior_json_fail:
                    print(
                        f"[warn] Window {window.window_id}: questionify JSON failure; skipping."
                    )
                    continue
                print(
                    f"[warn] Window {window.window_id}: unable to derive compliant question; skipping."
                )
                metrics["dropped_by_reason"]["question_refinement_fail"] += 1
                continue

            refined_question, context_text, context_speaker, origin, questionify_used = refinement
            if not questionify_used and origin in {"local", "context"}:
                metrics["preheuristic_questions"] += 1

            final_answer = base_answer
            answer_trim = apply_answer_trim(
                question=refined_question,
                original_answer=base_answer,
                config=answer_trim_config,
                trim_fn=answer_trim_fn,
            )

            pair_validation: Optional[Dict[str, Any]] = None
            validation_question: Optional[str] = None
            current_span_texts = list(base_span_texts)
            current_source_spans = "\n".join(current_span_texts)

            if answer_trim:
                # Phase 2: Validate contiguous substring before using trim
                if answer_trim.get("start_char", -1) == -1:
                    print(f"[warn] Window {window.window_id}: trim not contiguous substring; dropping.")
                    metrics["dropped_by_reason"]["trim_fail"] += 1
                    continue

                trimmed_answer = str(answer_trim.get("text") or "").strip()
                if trimmed_answer:
                    final_answer = trimmed_answer
                    current_span_texts = [trimmed_answer]
                    current_source_spans = "\n".join(current_span_texts)
                    pair["_answer_spans_text"] = current_span_texts
                    validation_question, pair_validation = evaluate_pairfit(
                        question=refined_question,
                        answer=final_answer,
                        source_spans=current_source_spans,
                        context_reference=context_text or current_source_spans,
                        config=pairfit_config,
                        pairfit_fn=pairfit_fn,
                    )
                    if validation_question is None:
                        status = (
                            pair_validation.get("status", "rejected")
                            if isinstance(pair_validation, dict)
                            else "rejected"
                        )
                        print(
                            f"[warn] Window {window.window_id}: trimmed answer validation {status}; reverting to original."
                        )
                        final_answer = base_answer
                        answer_trim = None
                        current_span_texts = list(base_span_texts)
                        current_source_spans = "\n".join(current_span_texts)
                        pair["_answer_spans_text"] = current_span_texts
                else:
                    answer_trim = None

            if validation_question is None:
                current_source_spans = "\n".join(current_span_texts)
                validation_question, pair_validation = evaluate_pairfit(
                    question=refined_question,
                    answer=final_answer,
                    source_spans=current_source_spans,
                    context_reference=context_text or current_source_spans,
                    config=pairfit_config,
                    pairfit_fn=pairfit_fn,
                )

            if validation_question is None or pair_validation is None:
                status = pair_validation.get("status", "rejected") if isinstance(pair_validation, dict) else "rejected"
                print(
                    f"[warn] Window {window.window_id}: pair validation {status}; skipping."
                )
                metrics["dropped_by_reason"]["pair_validation_fail"] += 1
                continue

            speaker_check_result = speaker_check_fn(speaker, final_answer)
            if speaker_check_result:
                if not speaker_check_result.get("allowed", False):
                    reason = speaker_check_result.get("reason") or "speaker not allowed"
                    print(
                        f"[warn] Window {window.window_id}: speaker compliance failed ({reason}); skipping."
                    )
                    metrics["dropped_by_reason"]["speaker_check_fail"] += 1
                    continue

            pair["question"] = validation_question
            pair["answer"] = final_answer
            current_source_spans = "\n".join(current_span_texts)
            if current_source_spans:
                pair["_source_spans"] = current_source_spans
            pair["_questionify_used"] = questionify_used
            pair["_answer_spans_text"] = list(current_span_texts)
            pair["_question_origin"] = origin
            if context_text:
                pair["_question_context"] = context_text
            if context_speaker:
                pair["_question_context_speaker"] = context_speaker
            if pair_validation:
                pair["_pair_validation"] = pair_validation
            if answer_trim:
                pair["_answer_trim"] = {**answer_trim, "threshold_tokens": answer_trim_config.threshold_tokens}
            if speaker_check_result:
                pair["_speaker_check"] = speaker_check_result

            speaker_check_status = "ok" if speaker_check_result else None

            # Track per-speaker counts
            if speaker not in metrics["per_speaker_counts"]:
                metrics["per_speaker_counts"][speaker] = 0
            metrics["per_speaker_counts"][speaker] += 1

            # Build HARMONY-STRICT training record and sidecar metadata
            try:
                training_record, sidecar_record = build_harmony_record(
                    pair,
                    window,
                    sample_idx=total_pairs,
                    composition_config=composition_config,
                )
            except ValueError as exc:
                logging.warning("Window %s: provenance/composition failure (%s); dropping pair.", window.window_id, exc)
                metrics["dropped_by_reason"]["provenance_fail"] += 1
                continue
            training_records.append(training_record)
            sidecar_metadata.append(sidecar_record)

            first_source = pair.get("citations", [{}])[0]
            source_ts = format_timestamp(first_source.get("start", math.nan))
            audit_rows.append(
                {
                    "episode": window.episode_id,
                    "window": window.window_id,
                    "question": pair["question"].replace("\n", " "),
                    "answer_speaker": speaker,
                    "speaker_check": speaker_check_status,
                    "source": source_ts,
                    "question_origin": origin,
                    "pairfit_status": pair_validation.get("status") if isinstance(pair_validation, dict) else None,
                    "qc_status": None,
                }
            )

        metrics["kept_pairs"] = len(training_records)
        metrics["total_pairs"] = total_pairs
        if time.time() - last_log_time >= 60:
            dropped_total = sum(metrics["dropped_by_reason"].values())
            kept = len(training_records)
            logging.info(
                "Progress windows_processed=%d/%d kept=%d dropped=%d | openai_calls=%d cache_hits=%d preheuristic_questions=%d | drop: questionify_json_fail=%d answer_not_found=%d pair_validation_fail=%d",
                metrics["windows_processed"],
                metrics["total_windows"],
                kept,
                dropped_total,
                metrics["openai_calls"],
                metrics["openai_cache_hits"],
                metrics["preheuristic_questions"],
                metrics["dropped_by_reason"]["questionify_json_fail"],
                metrics["dropped_by_reason"]["answer_not_found"],
                metrics["dropped_by_reason"]["pair_validation_fail"],
            )
            last_log_time = time.time()

    run_qc_audit(client, qc_config, training_records, audit_rows)
    if args.audit_path:
        write_audit(Path(args.audit_path), audit_rows)

    # Split records maintaining current train/val split
    train_records, val_records = split_records(training_records, args.train_split, args.seed)
    train_metadata, val_metadata = split_records(sidecar_metadata, args.train_split, args.seed)

    return (train_records, val_records), (train_metadata, val_metadata)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Generate Harmony-STRICT Q/A fine-tuning data from transcripts.")
    parser.add_argument("--input_glob", nargs="+", required=True, help="Transcript JSONL glob(s).")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory for Harmony JSONL output.")
    parser.add_argument(
        "--answer_speakers",
        type=str,
        required=True,
        help="Comma-separated list of speakers whose words may appear in answers.",
    )
    parser.add_argument("--model", type=str, default="gpt-4o", help="OpenAI model to use.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature.")
    parser.add_argument("--context_turns", type=int, default=6, help="Number of turns of context on each side.")
    parser.add_argument("--max_context_tokens", type=int, default=3200, help="Token cap for context windows.")
    parser.add_argument(
        "--window_stride",
        type=int,
        default=1,
        help="Advance transcript windows by this stride instead of 1.",
    )
    parser.add_argument("--questions_per_window", type=int, default=1, help="Max Q/A pairs per window.")
    parser.add_argument("--max_output_tokens", type=int, default=1200, help="Model response token limit.")
    parser.add_argument("--train_split", type=float, default=0.95, help="Fraction of samples for train.jsonl.")
    parser.add_argument("--seed", type=int, default=7, help="Shuffle seed for train/val split.")
    parser.add_argument("--audit_path", type=Path, help="Optional markdown audit file path.")
    parser.add_argument("--dry_run_windows", type=int, help="Limit number of windows for testing.")
    parser.add_argument(
        "--questionify_model",
        type=str,
        default="gpt-4o-mini",
        help="Model for rewriting non-interrogative context into questions (empty string to disable).",
    )
    parser.add_argument(
        "--questionify_temperature",
        type=float,
        default=0.2,
        help="Temperature for the question rewriting model.",
    )
    parser.add_argument(
        "--questionify_max_output_tokens",
        type=int,
        default=200,
        help="Token cap for the question rewriting model.",
    )
    parser.add_argument(
        "--max_openai_concurrency",
        type=int,
        default=8,
        help="Maximum concurrent OpenAI requests (questionify, trims, checks).",
    )
    parser.add_argument(
        "--questionify_cache_size",
        type=int,
        default=QUESTIONIFY_CACHE_CAPACITY_DEFAULT,
        help="LRU cache size for questionify prompts.",
    )
    parser.add_argument(
        "--questionify_guard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require minimum context length before invoking questionify.",
    )
    parser.add_argument(
        "--questionify_min_chars",
        type=int,
        default=30,
        help="Minimum characters of context required before questionify (guard enabled).",
    )
    parser.add_argument(
        "--disable_questionify",
        action="store_true",
        help="Skip rewriting non-interrogative context into explicit questions.",
    )
    parser.add_argument(
        "--answer_trim_model",
        type=str,
        default="gpt-4o-mini",
        help="Model for trimming long answers (empty string to disable).",
    )
    parser.add_argument(
        "--answer_trim_temperature",
        type=float,
        default=0.1,
        help="Temperature for the answer trimming model.",
    )
    parser.add_argument(
        "--answer_trim_max_output_tokens",
        type=int,
        default=160,
        help="Token cap for the answer trimming model.",
    )
    parser.add_argument(
        "--answer_trim_threshold",
        type=int,
        default=150,
        help="Token threshold above which answer trimming is attempted.",
    )
    parser.add_argument(
        "--disable_answer_trim",
        action="store_true",
        help="Skip the answer trimming pass.",
    )
    parser.add_argument(
        "--speaker_check_model",
        type=str,
        default="",
        help="Model for speaker allow-list compliance checks (empty string to disable).",
    )
    parser.add_argument(
        "--speaker_check_temperature",
        type=float,
        default=0.0,
        help="Temperature for the speaker compliance model.",
    )
    parser.add_argument(
        "--speaker_check_max_output_tokens",
        type=int,
        default=80,
        help="Token cap for the speaker compliance model.",
    )
    parser.add_argument(
        "--disable_speaker_check",
        action="store_true",
        help="Skip the speaker compliance prompt.",
    )
    parser.add_argument(
        "--qc_model",
        type=str,
        default="gpt-4o",
        help="Model for random-sample QC auditing (empty string to disable).",
    )
    parser.add_argument(
        "--qc_temperature",
        type=float,
        default=0.0,
        help="Temperature for the QC auditing model.",
    )
    parser.add_argument(
        "--qc_max_output_tokens",
        type=int,
        default=480,
        help="Token cap for the QC auditing model.",
    )
    parser.add_argument(
        "--qc_sample_rate",
        type=float,
        default=0.02,
        help="Fraction of records to audit (0.0-1.0).",
    )
    parser.add_argument(
        "--qc_min_samples",
        type=int,
        default=1,
        help="Minimum number of records to audit.",
    )
    parser.add_argument(
        "--qc_pass_threshold",
        type=float,
        default=0.95,
        help="Required pass rate for the QC audit (0-1).",
    )
    parser.add_argument(
        "--qc_seed",
        type=int,
        default=73,
        help="Random seed for selecting QC samples.",
    )
    parser.add_argument(
        "--qc_output_path",
        type=Path,
        help="Optional JSON output path for QC failures.",
    )
    parser.add_argument(
        "--disable_qc_audit",
        action="store_true",
        help="Skip the random-sample QC audit.",
    )
    parser.add_argument(
        "--pairfit_model",
        type=str,
        default="gpt-4o-mini",
        help="Model for semantic pair validation (empty string to disable).",
    )
    parser.add_argument(
        "--pairfit_temperature",
        type=float,
        default=0.1,
        help="Temperature for the pair validation model.",
    )
    parser.add_argument(
        "--pairfit_max_output_tokens",
        type=int,
        default=220,
        help="Token cap for the pair validation model.",
    )
    parser.add_argument(
        "--ce_batch_size",
        type=int,
        default=256,
        help="Batch size for cross-encoder semantic scoring.",
    )
    parser.add_argument(
        "--ce_max_length",
        type=int,
        default=512,
        help="Maximum token length for cross-encoder inputs.",
    )
    parser.add_argument(
        "--disable_pairfit",
        action="store_true",
        help="Skip the semantic validation pass for generated pairs.",
    )
    parser.add_argument(
        "--variant_b_ratio",
        type=float,
        default=0.0,
        help="Fraction of pairs to emit as Variant B (light paraphrase).",
    )
    parser.add_argument(
        "--connector_max_sentences",
        type=int,
        default=1,
        help="Maximum number of connector sentences allowed in Variant B.",
    )
    parser.add_argument(
        "--paraphrase_char_cap",
        type=int,
        default=300,
        help="Maximum total characters of paraphrase allowed in Variant B answers.",
    )
    parser.add_argument(
        "--variant_seed",
        type=int,
        default=42,
        help="Deterministic seed for Variant A/B assignment.",
    )
    args = parser.parse_args()

    # Generate HARMONY-STRICT records
    (train_records, val_records), (train_metadata, val_metadata) = generate_records(args)

    # Write HARMONY-STRICT training files (ONLY messages)
    write_jsonl(args.out_dir / "train.jsonl", train_records)
    write_jsonl(args.out_dir / "val.jsonl", val_records)

    # Write sidecar metadata files (1:1 alignment with training files) - normalized naming
    write_jsonl(args.out_dir / "sidecar_train.jsonl", train_metadata)
    write_jsonl(args.out_dir / "sidecar_val.jsonl", val_metadata)

    # Back-compat: also write legacy names for this release
    write_jsonl(args.out_dir / "train_metadata.jsonl", train_metadata)
    write_jsonl(args.out_dir / "val_metadata.jsonl", val_metadata)

    print(f"[done] Wrote {len(train_records)} train and {len(val_records)} val Harmony-STRICT records to {args.out_dir}")
    print(f"[done] Wrote {len(train_metadata)} train and {len(val_metadata)} val sidecar metadata records")


if __name__ == "__main__":
    main()
