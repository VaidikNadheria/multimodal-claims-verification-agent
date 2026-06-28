"""CSV and path helpers for the evidence review pipeline."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, Mapping, Optional

from schema import REQUIRED_OUTPUT_COLUMNS

LOGGER = logging.getLogger(__name__)

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - pandas is optional at runtime.
    pd = None


def read_csv_rows(path: Path, required_columns: Optional[list[str]] = None) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    if pd is not None:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows = frame.to_dict(orient="records")
        fieldnames = list(frame.columns)
    else:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = [{key: value or "" for key, value in row.items()} for row in reader]
            fieldnames = reader.fieldnames or []

    if required_columns:
        missing = [column for column in required_columns if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
    return [{key: str(value) for key, value in row.items()} for row in rows]


def write_csv_rows(path: Path, rows: Iterable[Mapping[str, str]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def load_lookup(path: Path, key_column: str) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(path, required_columns=[key_column])
    return {row[key_column]: row for row in rows}


def load_evidence_requirements(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(
        path,
        required_columns=[
            "requirement_id",
            "claim_object",
            "applies_to",
            "minimum_image_evidence",
        ],
    )


def read_existing_output(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    rows = read_csv_rows(path, required_columns=REQUIRED_OUTPUT_COLUMNS)
    LOGGER.info("Loaded %s existing output rows for resume from %s", len(rows), path)
    return rows


def resolve_dataset_image(dataset_dir: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return dataset_dir / path

