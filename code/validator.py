"""Strict output validation and conservative normalization helpers."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from schema import (
    BOOL_VALUES,
    CLAIM_OBJECTS,
    CLAIM_STATUS_VALUES,
    INPUT_COLUMNS,
    ISSUE_TYPE_VALUES,
    OBJECT_PART_VALUES,
    PREDICTION_COLUMNS,
    REQUIRED_OUTPUT_COLUMNS,
    RISK_FLAG_ORDER,
    RISK_FLAG_VALUES,
    SEVERITY_VALUES,
)


class OutputValidationError(ValueError):
    pass


def validate_input_rows(rows: Sequence[Mapping[str, str]]) -> None:
    for index, row in enumerate(rows, start=1):
        missing = [column for column in INPUT_COLUMNS if column not in row]
        if missing:
            raise OutputValidationError(f"input row {index} missing columns {missing}")
        if row.get("claim_object") not in CLAIM_OBJECTS:
            raise OutputValidationError(
                f"input row {index} has invalid claim_object={row.get('claim_object')!r}"
            )


def build_output_row(
    claim: Mapping[str, str],
    prediction: Mapping[str, Any],
    valid_image_ids: Sequence[str],
) -> dict[str, str]:
    row = {column: str(claim.get(column, "")).strip() for column in INPUT_COLUMNS}
    row.update(_normalize_prediction(prediction, row.get("claim_object", ""), valid_image_ids))
    validate_output_row(row, valid_image_ids=valid_image_ids)
    return {column: row[column] for column in REQUIRED_OUTPUT_COLUMNS}


def normalization_issues(
    prediction: Mapping[str, Any],
    claim_object: str,
    valid_image_ids: Sequence[str],
) -> list[str]:
    """Return fields where raw model output would be normalized by validation."""

    normalized = _normalize_prediction(prediction, claim_object, valid_image_ids)
    issues: list[str] = []
    for column in PREDICTION_COLUMNS:
        raw = prediction.get(column)
        if column in {"evidence_standard_met", "valid_image"}:
            raw_text = _normalize_bool(raw, default="")
        elif column == "risk_flags":
            raw_text = _normalize_semicolon_values(raw, RISK_FLAG_VALUES, default="")
        elif column == "supporting_image_ids":
            raw_text = _normalize_supporting_image_ids(raw, valid_image_ids)
        elif column == "issue_type":
            raw_text = _normalize_allowed(raw, ISSUE_TYPE_VALUES, default="")
        elif column == "object_part":
            raw_text = _normalize_allowed(
                raw,
                OBJECT_PART_VALUES.get(claim_object, {"unknown"}),
                default="",
            )
        elif column == "claim_status":
            raw_text = _normalize_allowed(raw, CLAIM_STATUS_VALUES, default="")
        elif column == "severity":
            raw_text = _normalize_allowed(raw, SEVERITY_VALUES, default="")
        else:
            raw_text = _clean_text(raw, "")
        if raw_text != normalized[column]:
            issues.append(column)
    return issues


def validate_output_rows(rows: Sequence[Mapping[str, str]]) -> None:
    for index, row in enumerate(rows, start=1):
        validate_output_row(row, row_number=index)


def validate_output_row(
    row: Mapping[str, str],
    row_number: int | None = None,
    valid_image_ids: Sequence[str] | None = None,
) -> None:
    prefix = f"output row {row_number}" if row_number is not None else "output row"
    if list(row.keys()) != REQUIRED_OUTPUT_COLUMNS:
        raise OutputValidationError(
            f"{prefix} must have exact columns {REQUIRED_OUTPUT_COLUMNS}"
        )
    for column in REQUIRED_OUTPUT_COLUMNS:
        if row.get(column) is None or str(row.get(column, "")) == "":
            raise OutputValidationError(f"{prefix} has missing cell in {column}")

    if row["claim_object"] not in CLAIM_OBJECTS:
        raise OutputValidationError(f"{prefix} invalid claim_object={row['claim_object']!r}")
    if row["evidence_standard_met"] not in BOOL_VALUES:
        raise OutputValidationError(f"{prefix} invalid evidence_standard_met")
    if row["valid_image"] not in BOOL_VALUES:
        raise OutputValidationError(f"{prefix} invalid valid_image")
    if row["issue_type"] not in ISSUE_TYPE_VALUES:
        raise OutputValidationError(f"{prefix} invalid issue_type={row['issue_type']!r}")
    if row["object_part"] not in OBJECT_PART_VALUES[row["claim_object"]]:
        raise OutputValidationError(f"{prefix} invalid object_part={row['object_part']!r}")
    if row["claim_status"] not in CLAIM_STATUS_VALUES:
        raise OutputValidationError(f"{prefix} invalid claim_status={row['claim_status']!r}")
    if row["severity"] not in SEVERITY_VALUES:
        raise OutputValidationError(f"{prefix} invalid severity={row['severity']!r}")
    _validate_semicolon_values(prefix, "risk_flags", row["risk_flags"], RISK_FLAG_VALUES)
    _validate_supporting_images(prefix, row["supporting_image_ids"], valid_image_ids)


def _normalize_prediction(
    prediction: Mapping[str, Any],
    claim_object: str,
    valid_image_ids: Sequence[str],
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    normalized["evidence_standard_met"] = _normalize_bool(
        prediction.get("evidence_standard_met"), default="false"
    )
    normalized["evidence_standard_met_reason"] = _clean_text(
        prediction.get("evidence_standard_met_reason"),
        "Insufficient evidence to verify the claim automatically.",
    )
    normalized["risk_flags"] = _normalize_semicolon_values(
        prediction.get("risk_flags"), RISK_FLAG_VALUES, default="manual_review_required"
    )
    normalized["issue_type"] = _normalize_allowed(
        prediction.get("issue_type"), ISSUE_TYPE_VALUES, default="unknown"
    )
    normalized["object_part"] = _normalize_allowed(
        prediction.get("object_part"),
        OBJECT_PART_VALUES.get(claim_object, {"unknown"}),
        default="unknown",
    )
    normalized["claim_status"] = _normalize_allowed(
        prediction.get("claim_status"),
        CLAIM_STATUS_VALUES,
        default="not_enough_information",
    )
    normalized["claim_status_justification"] = _clean_text(
        prediction.get("claim_status_justification"),
        "The automated pipeline could not verify the visual claim with high confidence.",
    )
    normalized["supporting_image_ids"] = _normalize_supporting_image_ids(
        prediction.get("supporting_image_ids"), valid_image_ids
    )
    normalized["valid_image"] = _normalize_bool(prediction.get("valid_image"), default="false")
    normalized["severity"] = _normalize_allowed(
        prediction.get("severity"), SEVERITY_VALUES, default="unknown"
    )
    return {column: normalized[column] for column in PREDICTION_COLUMNS}


def _normalize_bool(value: Any, default: str) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value).strip().lower()
    return text if text in BOOL_VALUES else default


def _normalize_allowed(value: Any, allowed: set[str], default: str) -> str:
    text = str(value).strip().lower()
    return text if text in allowed else default


def _normalize_semicolon_values(value: Any, allowed: set[str], default: str) -> str:
    if isinstance(value, list):
        parts = [str(item).strip().lower() for item in value]
    else:
        parts = [part.strip().lower() for part in str(value or "").split(";")]
    parts = [part for part in parts if part and part in allowed]
    if not parts or "none" in parts:
        return "none" if "none" in parts else default
    deduped = list(dict.fromkeys(part for part in parts if part != "none"))
    if allowed == RISK_FLAG_VALUES:
        ordered = [flag for flag in RISK_FLAG_ORDER if flag in deduped]
        extras = [flag for flag in deduped if flag not in ordered]
        deduped = ordered + extras
    return ";".join(deduped) if deduped else default


def _normalize_supporting_image_ids(value: Any, valid_image_ids: Sequence[str]) -> str:
    valid = set(valid_image_ids)
    if isinstance(value, list):
        parts = [str(item).strip() for item in value]
    else:
        parts = [part.strip() for part in str(value or "").split(";")]
    parts = [part for part in parts if part]
    if not parts or "none" in {part.lower() for part in parts}:
        return "none"
    if valid:
        parts = [part for part in parts if part in valid]
    parts = [part for part in parts if re.fullmatch(r"[A-Za-z0-9_-]+", part)]
    deduped = list(dict.fromkeys(parts))
    return ";".join(deduped) if deduped else "none"


def _clean_text(value: Any, default: str) -> str:
    text = " ".join(str(value or "").split())
    return text or default


def _validate_semicolon_values(prefix: str, column: str, value: str, allowed: set[str]) -> None:
    parts = value.split(";")
    if not parts or any(part == "" for part in parts):
        raise OutputValidationError(f"{prefix} invalid {column} formatting")
    if "none" in parts and len(parts) > 1:
        raise OutputValidationError(f"{prefix} cannot mix none with other {column}")
    invalid = [part for part in parts if part not in allowed]
    if invalid:
        raise OutputValidationError(f"{prefix} invalid {column} values {invalid}")


def _validate_supporting_images(
    prefix: str,
    value: str,
    valid_image_ids: Sequence[str] | None,
) -> None:
    parts = value.split(";")
    if not parts or any(part == "" for part in parts):
        raise OutputValidationError(f"{prefix} invalid supporting_image_ids formatting")
    if "none" in parts and len(parts) > 1:
        raise OutputValidationError(f"{prefix} cannot mix none with supporting image IDs")
    if parts == ["none"]:
        return
    if not all(re.fullmatch(r"[A-Za-z0-9_-]+", part) for part in parts):
        raise OutputValidationError(f"{prefix} invalid supporting_image_ids characters")
    if valid_image_ids is not None:
        valid = set(valid_image_ids)
        unknown = [part for part in parts if part not in valid]
        if unknown:
            raise OutputValidationError(
                f"{prefix} supporting_image_ids not in submitted images: {unknown}"
            )
