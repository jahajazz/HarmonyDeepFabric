"""Local question mining heuristics for Harmony generation."""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

INTERROGATIVE_PATTERN = re.compile(
    r"^\s*(who|what|why|how|where|when|which|whom|whose)\b", re.IGNORECASE
)
QUESTION_TOKEN_RE = re.compile(r"\?")
CUE_PHRASES = (
    "let me ask",
    "the question is",
    "the question then is",
    "what i am wondering",
    "are you saying",
    "would you say",
    "could it be that",
    "am i right that",
    "is it fair to say",
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\u2026])\s+")
WORD_RE = re.compile(r"\b[\w'-]+\b")


def _split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _ensure_question_mark(text: str) -> str:
    stripped = text.strip()
    if not stripped.endswith("?"):
        stripped += "?"
    return stripped


def _first_word(text: str) -> Optional[str]:
    match = WORD_RE.search(text.lower())
    return match.group(0) if match else None


def _iter_non_answer_turns(window: Any, max_turns: int, answer_speaker: str) -> List[Any]:
    if not getattr(window, "segments", None):
        return []
    limit = max(0, getattr(window, "anchor_index", 0))
    turns: List[Any] = []
    answer_lower = answer_speaker.lower()
    for seg in reversed(window.segments[:limit]):
        if seg.speaker.lower() == answer_lower:
            continue
        text = (seg.text or "").strip()
        if not text:
            continue
        turns.append(seg)
        if len(turns) >= max_turns:
            break
    return turns


def _clean_candidate(sentence: str) -> Optional[str]:
    candidate = sentence.strip()
    if not candidate:
        return None
    candidate = _ensure_question_mark(candidate)
    if _word_count(candidate) > 25:
        return None
    return candidate


def mine_local_question(window: Any, answer_speaker: str, max_turns: int = 3) -> Optional[str]:
    """Attempt to mine a local question from recent non-answer turns."""
    turns = _iter_non_answer_turns(window, max(max_turns, 1), answer_speaker)
    if not turns:
        return None

    # Pass 1: existing sentence containing '?'
    for seg in turns:
        for sentence in _split_sentences(seg.text):
            if QUESTION_TOKEN_RE.search(sentence):
                candidate = _clean_candidate(sentence)
                if candidate:
                    return candidate

    # Pass 2: interrogative starters
    for seg in turns:
        for sentence in _split_sentences(seg.text):
            if not INTERROGATIVE_PATTERN.match(sentence):
                continue
            candidate = _clean_candidate(sentence)
            if candidate:
                return candidate

    # Pass 3: cue phrases leading into a question
    for seg in turns:
        lowered = seg.text.lower()
        for cue in CUE_PHRASES:
            idx = lowered.find(cue)
            if idx == -1:
                continue
            tail = seg.text[idx + len(cue) :].strip(" :,-")
            if not tail:
                continue
            sentences = _split_sentences(tail)
            for sentence in sentences:
                candidate = _clean_candidate(sentence)
                if candidate:
                    return candidate

    return None


def last_non_answer_snippet(window: Any, max_turns: int, answer_speaker: str) -> Tuple[str, Optional[str]]:
    """Return the most recent non-answer text and its speaker."""
    turns = _iter_non_answer_turns(window, max_turns, answer_speaker)
    if not turns:
        return "", None
    seg = turns[0]
    return seg.text.strip(), seg.speaker
