"""Core pipeline wiring for multimodal evidence review."""

from __future__ import annotations

import json
import logging
from typing import Mapping

from adjudicator import Adjudicator
from config import RuntimeConfig
from gemini_client import GeminiVisionClient
from image_utils import collect_image_evidence
from io_utils import (
    load_evidence_requirements,
    load_lookup,
    read_csv_rows,
    read_existing_output,
    write_csv_rows,
)
from openai_client import OpenAIVisionClient
from prompts import PROMPT_VERSION, build_claim_prompt
from rules import (
    ADJUDICATION_MERGE_POLICY_VERSION,
    apply_identity_conflict_policy,
    apply_package_missing_content_policy,
    apply_taxonomy_calibration,
    apply_user_history_flag_policy,
    apply_wrong_object_taxonomy_policy,
    deterministic_fallback_prediction,
)
from schema import ClaimRow, INPUT_COLUMNS, REQUIRED_OUTPUT_COLUMNS
from validator import (
    build_output_row,
    normalization_issues,
    validate_input_rows,
    validate_output_rows,
)
from vlm_client import ModelResponseError, ModelUnavailable, VisionClient

LOGGER = logging.getLogger(__name__)


class ClaimPipeline:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.requirements = load_evidence_requirements(
            config.dataset_dir / "evidence_requirements.csv"
        )
        self.user_history = load_lookup(config.dataset_dir / "user_history.csv", "user_id")
        self.primary_client = self._build_client(config.primary_provider)
        self.fallback_client = self._build_client(config.fallback_provider)
        adjudication_client = None
        if config.adjudicate_suspicious and not config.no_adjudication and not config.dry_run:
            adjudication_client = self.fallback_client
        self.adjudicator = Adjudicator(adjudication_client)
        self._prior_adjudication_records = self._load_adjudication_records()

    def run(self) -> list[dict[str, str]]:
        rows = read_csv_rows(self.config.input_path, required_columns=INPUT_COLUMNS)
        if self.config.limit is not None:
            rows = rows[: self.config.limit]
        validate_input_rows(rows)

        claims = [ClaimRow.from_mapping(row) for row in rows]
        existing_by_key = self._load_resume_rows() if self.config.resume else {}
        retry_keys = self._failed_adjudication_keys(claims)
        retry_keys.update(self._stale_adjudication_policy_keys(claims))
        for key in retry_keys:
            existing_by_key.pop(key, None)
        if retry_keys:
            LOGGER.info(
                "Retrying %s rows whose prior adjudication did not complete",
                len(retry_keys),
            )
        output_rows: list[dict[str, str]] = []

        LOGGER.info(
            "Starting pipeline rows=%s dry_run=%s primary=%s fallback=%s",
            len(claims),
            self.config.dry_run,
            self.config.primary_provider,
            self.config.fallback_provider,
        )
        if self.config.dry_run:
            LOGGER.warning("Dry-run mode is active: no model calls will be made.")

        for index, claim in enumerate(claims, start=1):
            if claim.resume_key in existing_by_key:
                LOGGER.info("Reusing existing output row %s/%s", index, len(claims))
                output_rows.append(existing_by_key[claim.resume_key])
                continue

            output_row = self._process_claim(claim)
            output_rows.append(output_row)
            validate_output_rows(output_rows)
            write_csv_rows(self.config.output_path, output_rows, REQUIRED_OUTPUT_COLUMNS)

        validate_output_rows(output_rows)
        write_csv_rows(self.config.output_path, output_rows, REQUIRED_OUTPUT_COLUMNS)
        self._write_adjudication_metadata()
        LOGGER.info("Wrote %s rows to %s", len(output_rows), self.config.output_path)
        return output_rows

    def _process_claim(self, claim: ClaimRow) -> dict[str, str]:
        claim_dict = claim.to_input_dict()
        images = collect_image_evidence(self.config.dataset_dir, claim.image_paths)
        user_history = self.user_history.get(claim.user_id, {})
        image_ids = [image.image_id for image in images if image.exists]

        if self.config.dry_run:
            prediction = deterministic_fallback_prediction(
                claim_dict,
                images,
                user_history,
                reason="Dry-run mode did not call a vision model.",
            )
        else:
            prediction = self._model_prediction(claim_dict, images, user_history)
            prediction = apply_identity_conflict_policy(
                prediction,
                images,
                claim_object=claim.claim_object,
            )
            prediction = apply_wrong_object_taxonomy_policy(claim_dict, prediction)
            prediction = apply_taxonomy_calibration(prediction)
            prediction = apply_package_missing_content_policy(claim_dict, prediction)
            prediction = apply_user_history_flag_policy(prediction, user_history)

        normalized_fields = normalization_issues(
            prediction,
            claim.claim_object,
            valid_image_ids=image_ids,
        )
        extra_adjudication_reasons = [
            f"validator_normalized:{field}" for field in normalized_fields
        ]
        output_row = build_output_row(claim_dict, prediction, valid_image_ids=image_ids)
        if self.config.adjudicate_suspicious and not self.config.no_adjudication:
            output_row = self.adjudicator.maybe_adjudicate(
                claim=claim_dict,
                output_row=output_row,
                user_history=user_history,
                requirements=self.requirements,
                images=images,
                extra_reasons=extra_adjudication_reasons,
            )
        return output_row

    def _model_prediction(
        self,
        claim: Mapping[str, str],
        images,
        user_history: Mapping[str, str],
    ) -> dict:
        prompt = build_claim_prompt(
            claim=claim,
            user_history=user_history,
            requirements=self.requirements,
            images=images,
        )
        for client, label in [
            (self.primary_client, "primary"),
            (self.fallback_client, "fallback"),
        ]:
            if client is None:
                continue
            try:
                LOGGER.info("Calling %s provider=%s model=%s", label, client.provider, client.model)
                return client.generate_json(
                    prompt,
                    images,
                    cache_extra={
                        "pass": label,
                        "prompt_version": PROMPT_VERSION,
                        "claim": claim,
                        "user_history": user_history,
                        "evidence_requirements": self.requirements,
                    },
                )
            except (ModelUnavailable, ModelResponseError) as exc:
                LOGGER.warning("%s provider unavailable or failed: %s", label, exc)

        return deterministic_fallback_prediction(
            claim,
            images,
            user_history,
            reason="No configured vision provider returned a usable response.",
        )

    def _load_resume_rows(self) -> dict[str, dict[str, str]]:
        existing_rows = read_existing_output(self.config.output_path)
        if not existing_rows:
            return {}
        validate_output_rows(existing_rows)
        keyed = {}
        for row in existing_rows:
            claim = ClaimRow.from_mapping({column: row[column] for column in INPUT_COLUMNS})
            keyed[claim.resume_key] = {column: row[column] for column in REQUIRED_OUTPUT_COLUMNS}
        return keyed

    def _build_client(self, provider: str) -> VisionClient | None:
        provider = provider.lower()
        model = self.config.model_for_provider(provider)
        api_key = self.config.api_key_for_provider(provider)
        if provider == "gemini":
            return GeminiVisionClient(model=model, api_key=api_key, cache_dir=self.config.cache_dir)
        if provider == "openai":
            return OpenAIVisionClient(model=model, api_key=api_key, cache_dir=self.config.cache_dir)
        if provider in {"none", ""}:
            return None
        LOGGER.warning("Unknown provider %s; it will be ignored", provider)
        return None

    def _write_adjudication_metadata(self) -> None:
        records_by_key: dict[str, dict[str, object]] = {}
        for record in self._prior_adjudication_records:
            key = _adjudication_record_identity(record)
            if key:
                records_by_key[key] = record
        for record in self.adjudicator.records:
            key = _adjudication_record_identity(record)
            if key:
                records_by_key[key] = record

        if not records_by_key:
            return
        metadata_path = self.config.output_path.with_suffix(
            self.config.output_path.suffix + ".adjudication.jsonl"
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records_by_key.values():
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        LOGGER.info(
            "Wrote %s adjudication metadata rows to %s",
            len(records_by_key),
            metadata_path,
        )

    def _load_adjudication_records(self) -> list[dict[str, object]]:
        metadata_path = self.config.output_path.with_suffix(
            self.config.output_path.suffix + ".adjudication.jsonl"
        )
        if not metadata_path.exists():
            return []
        records: list[dict[str, object]] = []
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("Ignoring malformed adjudication metadata line")
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    def _failed_adjudication_keys(self, claims: list[ClaimRow]) -> set[str]:
        if not (
            self.config.resume
            and self.config.adjudicate_suspicious
            and not self.config.no_adjudication
        ):
            return set()
        failed = [
            record
            for record in self._prior_adjudication_records
            if record.get("status") == "failed"
        ]
        retry_keys: set[str] = set()
        for record in failed:
            key = str(record.get("claim_resume_key", ""))
            if key:
                retry_keys.add(key)
                continue
            matches = [
                claim.resume_key
                for claim in claims
                if claim.user_id == record.get("user_id", "")
                and claim.image_paths == record.get("image_paths", "")
                and claim.claim_object == record.get("claim_object", "")
            ]
            if len(matches) == 1:
                retry_keys.add(matches[0])
            elif matches:
                LOGGER.warning(
                    "Cannot safely resume failed adjudication with ambiguous claim identity"
                )
        return retry_keys

    def _stale_adjudication_policy_keys(self, claims: list[ClaimRow]) -> set[str]:
        """Reapply a new local merge policy without repeating model API calls."""

        if not (
            self.config.resume
            and self.config.adjudicate_suspicious
            and not self.config.no_adjudication
        ):
            return set()
        stale = [
            record
            for record in self._prior_adjudication_records
            if record.get("status") == "succeeded"
            and record.get("merge_policy_version") != ADJUDICATION_MERGE_POLICY_VERSION
        ]
        retry_keys: set[str] = set()
        for record in stale:
            key = str(record.get("claim_resume_key", ""))
            if key:
                retry_keys.add(key)
                continue
            matches = [
                claim.resume_key
                for claim in claims
                if claim.user_id == record.get("user_id", "")
                and claim.image_paths == record.get("image_paths", "")
                and claim.claim_object == record.get("claim_object", "")
            ]
            if len(matches) == 1:
                retry_keys.add(matches[0])
            elif matches:
                LOGGER.warning(
                    "Cannot safely reapply adjudication policy with ambiguous claim identity"
                )
        return retry_keys


def _adjudication_record_identity(record: Mapping[str, object]) -> str:
    """Stable sidecar identity that also works with metadata from older runs."""

    return "\u241f".join(
        str(record.get(column, "")).strip()
        for column in ("user_id", "image_paths", "claim_object")
    )
