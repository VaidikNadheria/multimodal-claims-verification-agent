"""Shared schemas and allowed values for the evidence review pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping


INPUT_COLUMNS: List[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
]

REQUIRED_OUTPUT_COLUMNS: List[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

PREDICTION_COLUMNS: List[str] = [
    column for column in REQUIRED_OUTPUT_COLUMNS if column not in INPUT_COLUMNS
]

CLAIM_OBJECTS = {"car", "laptop", "package"}

BOOL_VALUES = {"true", "false"}

CLAIM_STATUS_VALUES = {
    "supported",
    "contradicted",
    "not_enough_information",
}

ISSUE_TYPE_VALUES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}

OBJECT_PART_VALUES: Dict[str, set[str]] = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}

RISK_FLAG_ORDER: List[str] = [
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]

RISK_FLAG_VALUES = {"none", *RISK_FLAG_ORDER}

SEVERITY_VALUES = {"none", "low", "medium", "high", "unknown"}


@dataclass(frozen=True)
class ClaimRow:
    """Input claim row normalized to strings."""

    user_id: str
    image_paths: str
    user_claim: str
    claim_object: str

    @classmethod
    def from_mapping(cls, row: Mapping[str, object]) -> "ClaimRow":
        return cls(
            user_id=str(row.get("user_id", "")).strip(),
            image_paths=str(row.get("image_paths", "")).strip(),
            user_claim=str(row.get("user_claim", "")).strip(),
            claim_object=str(row.get("claim_object", "")).strip(),
        )

    @property
    def image_path_list(self) -> List[str]:
        return [part.strip() for part in self.image_paths.split(";") if part.strip()]

    @property
    def resume_key(self) -> str:
        return "\u241f".join(
            [self.user_id, self.image_paths, self.user_claim, self.claim_object]
        )

    def to_input_dict(self) -> dict[str, str]:
        return {
            "user_id": self.user_id,
            "image_paths": self.image_paths,
            "user_claim": self.user_claim,
            "claim_object": self.claim_object,
        }
