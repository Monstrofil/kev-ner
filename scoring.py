"""Per-field correctness, by the field's declared TYPE: ``text`` is OCR-tolerant fuzzy (ratio >= 0.85),
``date`` exact, anything else exact after casefold. Vendored from the scorer the generative baseline is
judged with, so every contender here is scored the same way.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_TEXT_THRESHOLD = 0.85
_STRIP_PUNCT = ".,;:!?\"'«»“”„`()[]{}<>-—–/\\|"


def _norm_text(s) -> str:
    """casefold + collapse whitespace + strip surrounding punctuation — the generic normal form for
    OCR-noisy free text (no language-specific homoglyph table)."""
    s = re.sub(r"\s+", " ", str(s if s is not None else "").casefold()).strip()
    return s.strip(_STRIP_PUNCT).strip()


def _text_match(gold: str, pred: str) -> bool:
    """OCR-tolerant fuzzy match: a char-level SequenceMatcher ratio over the normalised strings ≥ 0.85."""
    a, b = _norm_text(gold), _norm_text(pred)
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= _TEXT_THRESHOLD


def field_ok(field_type: str | None, gold, pred) -> bool:
    """True if the predicted field matches the gold under its TYPE's tolerance. Dispatch is on the
    declared TYPE, NOT the name: ``text`` → OCR-tolerant fuzzy (≥0.85); ``date`` → exact string (the
    model ISO-normalises per the instruction); ``number``/``enum``/default → exact after
    casefold+strip. ``null == null`` counts correct; ``null`` vs a value is wrong."""
    if gold is None and pred is None:
        return True
    if gold is None or pred is None:
        return False
    ftype = (field_type or "").strip().lower()
    if ftype == "text":
        return _text_match(gold, pred)
    if ftype == "date":
        return str(gold).strip() == str(pred).strip()
    return str(gold).strip().casefold() == str(pred).strip().casefold()
