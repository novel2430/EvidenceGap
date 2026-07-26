from __future__ import annotations

from enum import Enum

from evidencegap_backend.common import EvidenceGapError


class StanceLabel(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    INSUFFICIENT = "insufficient"


STANCE_LABELS = tuple(label.value for label in StanceLabel)


def canonical_stance_label(value: str | StanceLabel) -> str:
    label = value.value if isinstance(value, StanceLabel) else str(value).strip().lower()
    aliases = {
        "supported": StanceLabel.SUPPORT.value,
        "entailment": StanceLabel.SUPPORT.value,
        "refuted": StanceLabel.REFUTE.value,
        "contradiction": StanceLabel.REFUTE.value,
        "neutral": StanceLabel.INSUFFICIENT.value,
        "not_enough_information": StanceLabel.INSUFFICIENT.value,
        "nei": StanceLabel.INSUFFICIENT.value,
    }
    label = aliases.get(label, label)
    if label not in STANCE_LABELS:
        raise EvidenceGapError(
            f"Unknown stance label {value!r}; expected one of {STANCE_LABELS}"
        )
    return label
