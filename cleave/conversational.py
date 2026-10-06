"""Conversational and dialogue knowledge unit extraction.

Detects higher-level conversational patterns (question_answer, decision, action_item,
discussion) in dialogue transcripts and audio turns.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .models import ContentElement, KnowledgeUnitType

log = logging.getLogger(__name__)

_DECISION_PATTERNS = [
    re.compile(
        r"\b(we (agreed|decided|concluded)|decision (is|was)|let's (go with|move forward with|adopt)|consensus is|approved)\b",
        re.I,
    ),
]

_ACTION_PATTERNS = [
    re.compile(
        r"\b(action item|to-?do|will follow up|assign(ed)? to|take the lead on|action point)\b",
        re.I,
    ),
]

_QUESTION_PATTERNS = [
    re.compile(
        r"^(who|what|where|when|why|how|can|could|would|should|is|are|do|does|did)\b.*\?", re.I
    ),
    re.compile(r"\?$", re.M),
]


def classify_conversational_elements(
    elements: list[ContentElement],
) -> tuple[str, dict[str, Any]]:
    """Determine the KnowledgeUnitType and conversational metadata for a group of dialogue elements."""
    text = " ".join(e.text for e in elements)
    speakers = list(dict.fromkeys(e.speaker for e in elements if e.speaker))

    # 1. Action item detection
    if any(rx.search(text) for rx in _ACTION_PATTERNS):
        return KnowledgeUnitType.ACTION_ITEM.value, {
            "participants": speakers,
            "has_action_item": True,
        }

    # 2. Decision detection
    if any(rx.search(text) for rx in _DECISION_PATTERNS):
        return KnowledgeUnitType.DECISION.value, {
            "participants": speakers,
            "has_decision": True,
        }

    # 3. Question-Answer pairing
    has_question = (
        any(any(rx.search(e.text) for rx in _QUESTION_PATTERNS) for e in elements[:-1])
        if len(elements) > 1
        else False
    )
    if has_question and len(speakers) > 1:
        return KnowledgeUnitType.QUESTION_ANSWER.value, {
            "participants": speakers,
            "conversational_shape": "Q&A",
        }

    # 4. Multi-party discussion
    if len(speakers) > 1:
        return KnowledgeUnitType.DISCUSSION.value, {
            "participants": speakers,
            "turn_count": len(elements),
        }

    # 5. Single speaker turn
    if speakers:
        return KnowledgeUnitType.SPEAKER_TURN.value, {
            "speaker": speakers[0],
            "turn_count": len(elements),
        }

    return KnowledgeUnitType.NARRATIVE.value, {}
