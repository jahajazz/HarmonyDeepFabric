"""Cross-encoder utilities for Harmony pair-fit scoring."""

from __future__ import annotations

import logging
import math
from statistics import mean
from typing import List, Optional, Sequence, Tuple

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "sentence-transformers must be installed to use the cross-encoder scorer."
    ) from exc

_CE_MODEL: Optional[CrossEncoder] = None
_CE_DEVICE = "cuda"
_CE_BATCH_SIZE = 256
_CE_MAX_LENGTH = 512
_CE_LOGGED = False


def prepare_cross_encoder(batch_size: int = 256, max_length: int = 512) -> None:
    """Configure batch size / max length and warm the global model instance."""
    global _CE_BATCH_SIZE, _CE_MAX_LENGTH, _CE_DEVICE, _CE_MODEL
    _CE_BATCH_SIZE = max(1, batch_size)
    _CE_MAX_LENGTH = max(1, max_length)
    if torch is None or not torch.cuda.is_available():  # type: ignore[attr-defined]
        if _CE_DEVICE != "cpu":
            _CE_MODEL = None
        _CE_DEVICE = "cpu"
        logging.warning("CUDA unavailable; running cross-encoder on CPU.")
    else:
        if _CE_DEVICE != "cuda":
            _CE_MODEL = None
        _CE_DEVICE = "cuda"
    _ensure_model()
    _log_startup()


def _ensure_model() -> CrossEncoder:
    global _CE_MODEL
    if _CE_MODEL is None:
        _CE_MODEL = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=_CE_DEVICE,
            max_length=_CE_MAX_LENGTH,
        )
        _CE_MODEL.model.eval()
    return _CE_MODEL


def _log_startup() -> None:
    global _CE_LOGGED
    if _CE_LOGGED:
        return
    logging.info("CE device: %s", _CE_DEVICE)
    logging.info("CE batch_size: %d", _CE_BATCH_SIZE)
    _CE_LOGGED = True


def get_cross_encoder() -> Optional[CrossEncoder]:
    """Return the shared cross-encoder instance, initialising it if needed."""
    try:
        return _ensure_model()
    except Exception:  # pragma: no cover - propagate graceful failure
        logging.exception("Unable to load cross-encoder model.")
        return None


def score_pairs_ce(pairs: Sequence[Tuple[str, str]]) -> List[Optional[float]]:
    """Batch score (question, answer) pairs with the shared cross-encoder."""
    if not pairs:
        return []

    model = _ensure_model()
    if torch is None:
        logging.error("Torch unavailable; cannot compute cross-encoder scores.")
        return [None] * len(pairs)

    pairs_list: List[Tuple[str, str]] = [(q.strip(), a.strip()) for q, a in pairs]
    with torch.no_grad():  # type: ignore[attr-defined]
        try:
            scores = model.predict(
                pairs_list,
                batch_size=_CE_BATCH_SIZE,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover
            logging.exception("Cross-encoder scoring failed: %s", exc)
            return [None] * len(pairs)

    if hasattr(scores, "tolist"):
        raw_scores = scores.tolist()
    else:
        raw_scores = list(scores)

    results: List[Optional[float]] = []
    for idx in range(len(pairs_list)):
        raw = raw_scores[idx] if idx < len(raw_scores) else None
        if raw is None:
            results.append(None)
            continue
        try:
            prob = 1.0 / (1.0 + math.exp(-float(raw)))
            results.append(max(0.0, min(1.0, prob)))
        except Exception:  # pragma: no cover
            results.append(None)

    valid_scores = [score for score in results if score is not None]
    if valid_scores:
        logging.info(
            "Cross-encoder batch stats: min=%.3f mean=%.3f max=%.3f (n=%d)",
            min(valid_scores),
            mean(valid_scores),
            max(valid_scores),
            len(valid_scores),
        )
    else:
        logging.info("Cross-encoder batch stats: no valid scores (n=%d)", len(results))
    return results
