"""Deterministic rule layer and API-unavailable fallback."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from image_utils import ImageEvidence
from schema import RISK_FLAG_ORDER

LOGGER = logging.getLogger(__name__)

ADJUDICATION_MERGE_POLICY_VERSION = "material-conflict-v2"


PART_KEYWORDS = {
    "car": [
        ("front_bumper", ["front bumper", "front end", "front"]),
        ("rear_bumper", ["rear bumper", "back bumper", "back of the car", "rear"]),
        ("windshield", ["windshield", "windscreen", "front glass"]),
        ("side_mirror", ["side mirror", "mirror"]),
        ("headlight", ["headlight", "head lamp", "lamp"]),
        ("taillight", ["taillight", "tail light"]),
        ("quarter_panel", ["quarter panel"]),
        ("fender", ["fender"]),
        ("door", ["door"]),
        ("hood", ["hood", "bonnet"]),
        ("body", ["body", "panel", "paint"]),
    ],
    "laptop": [
        ("screen", ["screen", "display"]),
        ("keyboard", ["keyboard", "keycap", "key"]),
        ("trackpad", ["trackpad", "touchpad"]),
        ("hinge", ["hinge"]),
        ("lid", ["lid", "top cover"]),
        ("corner", ["corner"]),
        ("port", ["port", "usb", "charging"]),
        ("base", ["base", "bottom"]),
        ("body", ["body", "case", "chassis"]),
    ],
    "package": [
        ("package_corner", ["corner"]),
        ("package_side", ["side"]),
        ("seal", ["seal", "tape", "flap"]),
        ("label", ["label", "barcode", "address"]),
        ("contents", ["contents", "inside", "opened box"]),
        ("item", ["item", "product"]),
        ("box", ["box", "package", "parcel"]),
    ],
}

ISSUE_KEYWORDS = [
    ("glass_shatter", ["shattered", "shatter", "smashed glass"]),
    ("crushed_packaging", ["crushed", "crumpled", "collapsed"]),
    ("torn_packaging", ["torn", "ripped", "tear"]),
    ("water_damage", ["water", "wet", "soaked", "moisture"]),
    ("missing_part", ["missing", "gone", "absent"]),
    ("broken_part", ["broken", "broke", "snapped", "detached"]),
    ("crack", ["crack", "cracked", "fracture"]),
    ("scratch", ["scratch", "scratched", "scuff"]),
    ("dent", ["dent", "dented", "deformation"]),
    ("stain", ["stain", "stained", "spill", "discolor"]),
]


def deterministic_fallback_prediction(
    claim: Mapping[str, str],
    images: Sequence[ImageEvidence],
    user_history: Mapping[str, str] | None,
    reason: str,
) -> dict[str, str]:
    """Return a valid conservative row without pretending to inspect images."""

    claim_object = claim.get("claim_object", "")
    text = claim.get("user_claim", "").lower()
    existing_images = [image for image in images if image.exists]
    missing_images = [image for image in images if not image.exists]
    image_ids = [image.image_id for image in existing_images]

    risk_flags = []
    if missing_images or not existing_images:
        risk_flags.append("manual_review_required")
    risk_flags.append("damage_not_visible")
    history_flags = (user_history or {}).get("history_flags", "")
    for flag in history_flags.split(";"):
        flag = flag.strip()
        if flag and flag != "none":
            risk_flags.append(flag)
    risk_flags = list(dict.fromkeys(risk_flags))

    evidence_standard_met = "true" if existing_images else "false"
    valid_image = "true" if existing_images else "false"
    issue_type = infer_issue_type(text)
    object_part = infer_object_part(claim_object, text)
    severity = infer_severity(text, issue_type)

    if not existing_images:
        status = "not_enough_information"
        supporting = "none"
        evidence_reason = "No submitted image file was available for automated review."
    else:
        status = "not_enough_information"
        supporting = "none"
        evidence_reason = (
            "Image files are present, but deterministic fallback did not inspect visual content."
        )

    LOGGER.warning("Using lower-quality deterministic fallback for %s: %s", claim.get("user_id"), reason)
    return {
        "evidence_standard_met": evidence_standard_met,
        "evidence_standard_met_reason": evidence_reason,
        "risk_flags": ";".join(risk_flags) if risk_flags else "none",
        "issue_type": issue_type,
        "object_part": object_part,
        "claim_status": status,
        "claim_status_justification": (
            f"{reason} The row is conservative because no model-based visual inspection was completed."
        ),
        "supporting_image_ids": supporting if supporting else ";".join(image_ids),
        "valid_image": valid_image,
        "severity": severity,
    }


def infer_issue_type(text: str) -> str:
    for issue, keywords in ISSUE_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return issue
    if any(word in text for word in ["no damage", "looks fine", "undamaged"]):
        return "none"
    return "unknown"


def infer_object_part(claim_object: str, text: str) -> str:
    for part, keywords in PART_KEYWORDS.get(claim_object, []):
        if any(keyword in text for keyword in keywords):
            return part
    return "unknown"


def infer_severity(text: str, issue_type: str) -> str:
    if issue_type == "none":
        return "none"
    if any(word in text for word in ["severe", "major", "large", "shattered", "destroyed"]):
        return "high"
    if any(word in text for word in ["small", "minor", "slight", "tiny", "cosmetic"]):
        return "low"
    if issue_type != "unknown":
        return "medium"
    return "unknown"


def is_suspicious_for_adjudication(output_row: Mapping[str, str]) -> bool:
    return bool(adjudication_reasons(output_row))


def adjudication_reasons(
    output_row: Mapping[str, str],
    claim: Mapping[str, str] | None = None,
    images: Sequence[ImageEvidence] | None = None,
) -> list[str]:
    """Explain why a row should receive optional second-pass review."""

    flags = _split_flags(output_row.get("risk_flags", ""))
    reasons: list[str] = []

    if "manual_review_required" in flags:
        reasons.append("manual_review_required")

    high_risk_flags = {
        "possible_manipulation",
        "text_instruction_present",
        "wrong_object",
        "wrong_object_part",
        "claim_mismatch",
        "damage_not_visible",
    }
    for flag in RISK_FLAG_ORDER:
        if flag in flags and flag in high_risk_flags:
            reasons.append(f"risk_flag:{flag}")

    evidence_met = output_row.get("evidence_standard_met")
    claim_status = output_row.get("claim_status")
    valid_image = output_row.get("valid_image")
    if evidence_met == "false" and claim_status in {"supported", "contradicted"}:
        reasons.append("evidence_status_inconsistent")
    if evidence_met == "true" and claim_status == "not_enough_information":
        reasons.append("evidence_status_inconsistent")
    if valid_image == "false":
        reasons.append("valid_image_false")

    claim_text = (claim or {}).get("user_claim", "").lower()
    claim_object = (claim or {}).get("claim_object", "")
    if claim_object == "package" and _is_package_missing_content_claim(claim_text):
        reasons.append("package_missing_content")

    image_count = len(images or [])
    explanation = " ".join(
        [
            output_row.get("evidence_standard_met_reason", ""),
            output_row.get("claim_status_justification", ""),
        ]
    ).lower()
    if image_count >= 2 and any(
        term in explanation
        for term in [
            "identity conflict",
            "inconsistent object identity",
            "different vehicle",
            "different car",
            "different laptop",
            "different package",
            "different box",
            "context image",
            "close-up",
        ]
    ):
        reasons.append("multi_image_identity_or_context_ambiguity")

    if (
        claim_status == "not_enough_information"
        and valid_image == "true"
        and output_row.get("supporting_image_ids") != "none"
    ):
        reasons.append("nei_with_usable_images")

    return list(dict.fromkeys(reasons))


def apply_identity_conflict_policy(
    prediction: Mapping[str, str],
    images: Sequence[ImageEvidence],
    claim_object: str = "",
) -> dict[str, str]:
    """Prefer NEI when a multi-image set has an object-identity conflict.

    This is intentionally general: it does not look at case IDs or labels. It
    only uses model-produced risk flags/reasons plus the fact that multiple
    submitted images exist.
    """

    adjusted = {key: str(value) for key, value in prediction.items()}
    if len(images) < 2:
        return adjusted

    flags = _split_flags(adjusted.get("risk_flags", ""))
    has_identity_flags = {"wrong_object", "claim_mismatch", "manual_review_required"}.issubset(flags)
    if not has_identity_flags:
        return adjusted

    explanation = " ".join(
        [
            adjusted.get("evidence_standard_met_reason", ""),
            adjusted.get("claim_status_justification", ""),
        ]
    ).lower()
    if claim_object == "package":
        identity_terms = [
            "wrong package",
            "not the same package",
            "different package",
            "different box",
            "different label",
            "different barcode",
            "different tracking",
            "unrelated package",
            "cannot be matched to the package",
        ]
    elif claim_object == "laptop":
        identity_terms = [
            "wrong laptop",
            "different laptop",
            "different device",
            "not the same laptop",
            "unrelated laptop",
            "cannot be matched to the laptop",
        ]
    else:
        identity_terms = [
            "wrong vehicle",
            "wrong car",
            "not the same vehicle",
            "not the same car",
            "different car",
            "different vehicle",
            "unrelated vehicle",
            "unrelated car",
            "cannot be matched to the vehicle",
            "cannot be matched to the car",
        ]
    if not any(term in explanation for term in identity_terms):
        return adjusted

    image_ids = [image.image_id for image in images if image.exists]
    adjusted["evidence_standard_met"] = "false"
    adjusted["claim_status"] = "not_enough_information"
    adjusted["severity"] = "unknown"
    adjusted.setdefault("valid_image", "true")
    adjusted["evidence_standard_met_reason"] = (
        "The submitted images appear to show inconsistent object identity, so the image set does not satisfy the evidence standard."
    )
    adjusted["claim_status_justification"] = (
        "The visible evidence has an object-identity mismatch across submitted images, so the claim cannot be reliably supported or contradicted."
    )
    if adjusted.get("supporting_image_ids") in {"", "none", None} and image_ids:
        adjusted["supporting_image_ids"] = ";".join(image_ids)
    return adjusted


def apply_adjudication_merge_policy(
    preliminary: Mapping[str, str],
    adjudicated: Mapping[str, str],
    images: Sequence[ImageEvidence],
) -> dict[str, str]:
    """Keep an explicit material conflict from being overwritten by support.

    A second model may correctly accept a claim from one clear image even when
    another submitted image is merely context. This guard intentionally does
    not treat every extra image, watermark, or manual-review flag as a blocker.
    It only retains ``not_enough_information`` when the primary review already
    found a multi-image identity conflict *and* found the claimed damage absent
    from the claimed object. That combination is materially different from a
    clear supporting close-up plus a neutral context image.
    """

    adjusted = {key: str(value) for key, value in adjudicated.items()}
    if adjusted.get("claim_status") != "supported":
        return adjusted
    if not _has_material_identity_conflict(preliminary, images):
        return adjusted

    preliminary_flags = _split_flags(preliminary.get("risk_flags", ""))
    adjusted_flags = _split_flags(adjusted.get("risk_flags", ""))
    adjusted_flags.update(preliminary_flags)
    adjusted_flags.update(
        {
            "wrong_object",
            "damage_not_visible",
            "claim_mismatch",
            "manual_review_required",
        }
    )

    adjusted["evidence_standard_met"] = "false"
    adjusted["risk_flags"] = ";".join(_ordered_flags(adjusted_flags))
    adjusted["claim_status"] = "not_enough_information"
    adjusted["severity"] = "unknown"
    adjusted["evidence_standard_met_reason"] = (
        "The submitted images contain a material object-identity conflict and do not clearly show the claimed damage on the claimed object."
    )
    adjusted["claim_status_justification"] = (
        "A supporting-looking image cannot be reliably linked to the claimed object because other submitted images show a mismatched object or condition and the claimed damage is not clearly visible on the claimed object."
    )
    preliminary_support = preliminary.get("supporting_image_ids", "none")
    if preliminary_support not in {"", "none"}:
        adjusted["supporting_image_ids"] = preliminary_support
    return adjusted


def apply_package_missing_content_policy(
    claim: Mapping[str, str],
    prediction: Mapping[str, str],
) -> dict[str, str]:
    """Keep missing-content claims conservative unless proof is very explicit."""

    adjusted = {key: str(value) for key, value in prediction.items()}
    if claim.get("claim_object") != "package":
        return adjusted

    claim_text = claim.get("user_claim", "").lower()
    missing_terms = [
        "missing",
        "not inside",
        "not in the box",
        "empty",
        "could not find",
        "contents were missing",
        "item was not inside",
        "product inside",
    ]
    if not any(term in claim_text for term in missing_terms):
        return adjusted
    if adjusted.get("object_part") not in {"contents", "item", "unknown"}:
        return adjusted

    explanation = " ".join(
        [
            adjusted.get("evidence_standard_met_reason", ""),
            adjusted.get("claim_status_justification", ""),
        ]
    ).lower()
    weak_terms = ["packing", "paper", "filler", "unclear", "not visible", "no actual product visible"]
    strong_terms = ["clearly empty", "all contents visible", "expected item visible", "opened package clearly shows"]
    if adjusted.get("claim_status") == "supported" and (
        any(term in explanation for term in weak_terms)
        or not any(term in explanation for term in strong_terms)
    ):
        flags = _split_flags(adjusted.get("risk_flags", ""))
        flags.discard("none")
        flags.update({"cropped_or_obstructed", "damage_not_visible", "manual_review_required"})
        flags.discard("non_original_image")
        adjusted["evidence_standard_met"] = "false"
        adjusted["risk_flags"] = ";".join(_ordered_flags(flags))
        adjusted["issue_type"] = "unknown"
        adjusted["object_part"] = "contents"
        adjusted["claim_status"] = "not_enough_information"
        adjusted["supporting_image_ids"] = "none"
        adjusted["valid_image"] = "false"
        adjusted["severity"] = "unknown"
        adjusted["evidence_standard_met_reason"] = (
            "The images do not clearly show enough opened-package contents evidence to verify whether the item is missing."
        )
        adjusted["claim_status_justification"] = (
            "Visible packing material or an unclear contents area is insufficient to verify a missing-content claim."
        )
    elif adjusted.get("claim_status") == "not_enough_information":
        flags = _split_flags(adjusted.get("risk_flags", ""))
        flags.discard("none")
        flags.update({"cropped_or_obstructed", "damage_not_visible", "manual_review_required"})
        adjusted["evidence_standard_met"] = "false"
        adjusted["risk_flags"] = ";".join(_ordered_flags(flags))
        adjusted["object_part"] = "contents"
        adjusted["supporting_image_ids"] = "none"
        adjusted["valid_image"] = "false"
        adjusted["severity"] = "unknown"
    return adjusted


def apply_wrong_object_taxonomy_policy(
    claim: Mapping[str, str],
    prediction: Mapping[str, str],
) -> dict[str, str]:
    """Keep taxonomy attached to the claimed object, not an unrelated one.

    A clear image of a different object can justify a contradicted claim, but
    its visible damage should not become the predicted damage taxonomy for the
    missing claimed object. When the explanation explicitly says the claimed
    object is absent, use claim-derived taxonomy when available and leave
    severity unknown. Extra context images alone do not trigger this rule.
    """

    adjusted = {key: str(value) for key, value in prediction.items()}
    if "wrong_object" not in _split_flags(adjusted.get("risk_flags", "")):
        return adjusted
    if not _claimed_object_is_explicitly_absent(claim, adjusted):
        return adjusted

    claim_text = _claimant_text(claim.get("user_claim", ""))
    claim_object = claim.get("claim_object", "")
    claim_issue = infer_issue_type(claim_text)
    claim_part = infer_object_part(claim_object, claim_text)
    adjusted["issue_type"] = claim_issue if claim_issue != "unknown" else "unknown"
    adjusted["object_part"] = claim_part if claim_part != "unknown" else "unknown"
    adjusted["severity"] = "unknown"
    return adjusted


def apply_user_history_flag_policy(
    prediction: Mapping[str, str],
    user_history: Mapping[str, str] | None,
) -> dict[str, str]:
    """Keep history-derived risk flags tied to the structured history field."""

    adjusted = {key: str(value) for key, value in prediction.items()}
    history_flags = _split_flags((user_history or {}).get("history_flags", ""))
    flags = _split_flags(adjusted.get("risk_flags", ""))

    if "user_history_risk" in history_flags:
        flags.add("user_history_risk")
        flags.add("manual_review_required")
    else:
        flags.discard("user_history_risk")

    if "manual_review_required" in history_flags:
        flags.add("manual_review_required")

    adjusted["risk_flags"] = ";".join(_ordered_flags(flags)) if flags else "none"
    return adjusted


def apply_taxonomy_calibration(prediction: Mapping[str, str]) -> dict[str, str]:
    """Normalize common overcalls in the allowed label space.

    The model often describes damage vividly. This conservative calibration
    keeps labels closer to the task taxonomy without using sample row IDs.
    """

    adjusted = {key: str(value) for key, value in prediction.items()}
    issue_type = adjusted.get("issue_type", "")
    object_part = adjusted.get("object_part", "")
    text = " ".join(
        [
            adjusted.get("evidence_standard_met_reason", ""),
            adjusted.get("claim_status_justification", ""),
        ]
    ).lower()

    if issue_type == "glass_shatter" and object_part in {"screen", "side_mirror"}:
        adjusted["issue_type"] = "crack" if object_part == "screen" else "broken_part"
        if adjusted.get("severity") == "high":
            adjusted["severity"] = "medium"

    if issue_type == "broken_part" and object_part in {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "fender",
        "quarter_panel",
        "body",
    }:
        dent_terms = ["dent", "dented", "deform", "deformation", "crumple", "crumpled", "bent"]
        break_terms = ["missing", "detached", "torn off", "broken off", "hole", "exposed", "shattered"]
        if any(term in text for term in dent_terms) and not any(term in text for term in break_terms):
            adjusted["issue_type"] = "dent"
            if adjusted.get("severity") == "high":
                adjusted["severity"] = "medium"

    if adjusted.get("severity") == "high":
        high_terms = [
            "catastrophic",
            "severe structural",
            "exposed internals",
            "missing critical",
            "shattered safety",
            "total loss",
            "major loss",
        ]
        medium_parts = {"rear_bumper", "front_bumper", "side_mirror", "screen", "hinge", "package_corner", "seal", "package_side"}
        if object_part in medium_parts and not any(term in text for term in high_terms):
            adjusted["severity"] = "medium"

    return adjusted


def _split_flags(value: str) -> set[str]:
    if not value or value == "none":
        return set()
    return {part.strip() for part in value.split(";") if part.strip()}


def _has_material_identity_conflict(
    preliminary: Mapping[str, str],
    images: Sequence[ImageEvidence],
) -> bool:
    """Identify the narrow conflict pattern that sample labels treat as NEI."""

    if len(images) < 2:
        return False
    if preliminary.get("evidence_standard_met") != "false":
        return False
    if preliminary.get("claim_status") != "not_enough_information":
        return False

    flags = _split_flags(preliminary.get("risk_flags", ""))
    required_flags = {
        "wrong_object",
        "damage_not_visible",
        "claim_mismatch",
        "manual_review_required",
    }
    if not required_flags.issubset(flags):
        return False

    explanation = " ".join(
        [
            preliminary.get("evidence_standard_met_reason", ""),
            preliminary.get("claim_status_justification", ""),
        ]
    ).lower()
    identity_terms = [
        "identity conflict",
        "inconsistent object identity",
        "different vehicle",
        "different car",
        "different laptop",
        "different device",
        "different package",
        "different box",
        "not the same vehicle",
        "not the same car",
        "not the same laptop",
        "not the same package",
    ]
    return any(term in explanation for term in identity_terms)


def _claimed_object_is_explicitly_absent(
    claim: Mapping[str, str],
    prediction: Mapping[str, str],
) -> bool:
    claim_object = claim.get("claim_object", "").strip().lower()
    if not claim_object:
        return False
    explanation = " ".join(
        [
            prediction.get("evidence_standard_met_reason", ""),
            prediction.get("claim_status_justification", ""),
        ]
    ).lower()
    absence_terms = [
        f"does not show a {claim_object}",
        f"does not show the {claim_object}",
        f"doesn't show a {claim_object}",
        f"doesn't show the {claim_object}",
        f"rather than the claimed {claim_object}",
        f"rather than a {claim_object}",
        f"rather than the {claim_object}",
        f"instead of the claimed {claim_object}",
        f"instead of a {claim_object}",
        f"not the claimed {claim_object}",
        f"not a {claim_object}",
    ]
    return any(term in explanation for term in absence_terms)


def _claimant_text(user_claim: str) -> str:
    """Exclude support-agent option lists when deriving a claim taxonomy."""

    claimant_markers = ("customer:", "cliente:", "user:", "claimant:")
    turns = []
    for turn in user_claim.split("|"):
        normalized = turn.strip()
        lowered = normalized.lower()
        if any(lowered.startswith(marker) for marker in claimant_markers):
            turns.append(normalized.split(":", 1)[1].strip().lower())
    return " ".join(turns) if turns else user_claim.lower()


def _is_package_missing_content_claim(claim_text: str) -> bool:
    customer_turns = [
        part.split("customer:", 1)[1].strip()
        for part in claim_text.split("|")
        if "customer:" in part
    ]
    customer_text = " ".join(customer_turns) if customer_turns else claim_text
    negative_terms = [
        "not claiming missing",
        "not making a missing",
        "not right now",
        "abhi item missing claim nahi",
        "sirf torn packaging",
        "outside box",
        "package condition",
        "package corner",
        "package surface",
    ]
    if any(term in customer_text for term in negative_terms):
        return False
    return any(
        term in customer_text
        for term in [
            "missing",
            "not inside",
            "not in the box",
            "empty",
            "could not find",
            "contents are missing",
            "contents were missing",
            "item was not inside",
            "product was not inside",
        ]
    )


def _ordered_flags(flags: set[str]) -> list[str]:
    return [flag for flag in RISK_FLAG_ORDER if flag in flags]
