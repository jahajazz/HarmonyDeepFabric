"""Utilities for strict verbatim span matching without LLM calls."""

from __future__ import annotations

import re
import unicodedata
from typing import List, Optional, Tuple

QUOTE_TRANSLATE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2032": "'",
        "\u2033": '"',
    }
)

TRIM_CHARS = set('.,!?;:"\'')


def _normalize_with_map(text: str) -> Tuple[str, List[int], Optional[int], Optional[int]]:
    chars: List[str] = []
    index_map: List[int] = []
    prev_space = False

    for idx, raw_char in enumerate(text):
        nfkc = unicodedata.normalize("NFKC", raw_char)
        for sub_char in nfkc:
            translated = sub_char.translate(QUOTE_TRANSLATE).lower()
            if translated == "\u2026":  # ellipsis char
                translated = "..."
            for token in translated:
                if token.isspace():
                    if prev_space:
                        continue
                    chars.append(" ")
                    index_map.append(idx)
                    prev_space = True
                else:
                    prev_space = False
                    chars.append(token)
                    index_map.append(idx)

    # Trim leading/trailing spaces
    start = 0
    end = len(chars)
    while start < end and chars[start] == " ":
        start += 1
    while end > start and chars[end - 1] == " ":
        end -= 1

    # Trim punctuation/ellipsis runs at boundaries
    while start < end and (chars[start] in TRIM_CHARS or chars[start] == "."):
        start += 1
        while start < end and chars[start] == " ":
            start += 1
    while end > start and (chars[end - 1] in TRIM_CHARS or chars[end - 1] == "."):
        end -= 1
        while end > start and chars[end - 1] == " ":
            end -= 1

    if start >= end:
        return "", [], None, None

    normalized = "".join(chars[start:end])
    trimmed_map = index_map[start:end]
    original_start = trimmed_map[0] if trimmed_map else None
    original_end = trimmed_map[-1] if trimmed_map else None
    return normalized, trimmed_map, original_start, original_end


def normalize(text: str) -> str:
    """Return normalized text following Harmony span matching rules."""
    normalized, _, _, _ = _normalize_with_map(text)
    return normalized


def find_verbatim_span(source: str, answer: str) -> Optional[Tuple[str, int, int]]:
    """Return the exact substring from source matching answer with strict rules."""
    if not answer.strip():
        return None
    if "..." in answer or "\u2026" in answer:
        return None

    norm_answer, answer_map, answer_start, answer_end = _normalize_with_map(answer)
    if not norm_answer or answer_start is None or answer_end is None:
        return None

    norm_source, source_map, _, _ = _normalize_with_map(source)
    idx = norm_source.find(norm_answer)
    if idx == -1:
        return None

    source_start = source_map[idx]
    source_end = source_map[idx + len(norm_answer) - 1] + 1
    source_slice = source[source_start:source_end]
    answer_slice = answer[answer_start:answer_end + 1]

    if source_slice != answer_slice:
        return None

    # Reject spans that terminate mid-word
    if source_slice and source_slice[-1].isalnum() and source_end < len(source) and source[source_end:source_end + 1].isalnum():
        return None
    if source_slice and source_slice[0].isalnum() and source_start > 0 and source[source_start - 1:source_start].isalnum():
        return None

    return source_slice, source_start, source_end


def verbatim_contains(source: str, answer: str) -> Optional[str]:
    """Compatibility wrapper returning the matched substring when present."""
    result = find_verbatim_span(source, answer)
    if result is None:
        return None
    return result[0]
