"""Optional second-pass adjudication for suspicious claims."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

from image_utils import ImageEvidence
from prompts import PROMPT_VERSION, build_adjudication_prompt
from rules import (
    ADJUDICATION_MERGE_POLICY_VERSION,
    adjudication_reasons,
    apply_adjudication_merge_policy,
    apply_identity_conflict_policy,
    apply_package_missing_content_policy,
    apply_taxonomy_calibration,
    apply_user_history_flag_policy,
    apply_wrong_object_taxonomy_policy,
)
from validator import build_output_row
from vlm_client import ModelResponseError, ModelUnavailable, VisionClient

LOGGER = logging.getLogger(__name__)


class Adjudicator:
    def __init__(self, client: VisionClient | None) -> None:
        self.client = client
        self.records: list[dict[str, object]] = []

    def maybe_adjudicate(
        self,
        claim: Mapping[str, str],
        output_row: Mapping[str, str],
        user_history: Mapping[str, str] | None,
        requirements: Sequence[Mapping[str, str]],
        images: Sequence[ImageEvidence],
        extra_reasons: Sequence[str] | None = None,
    ) -> dict[str, str]:
        reasons = adjudication_reasons(output_row, claim=claim, images=images)
        for reason in extra_reasons or []:
            if reason not in reasons:
                reasons.append(reason)

        if self.client is None or not reasons:
            return dict(output_row)

        image_ids = [image.image_id for image in images if image.exists]
        prompt = build_adjudication_prompt(
            claim=claim,
            preliminary_output=output_row,
            user_history=user_history,
            requirements=requirements,
            images=images,
        )
        record: dict[str, object] = {
            "claim_resume_key": _claim_resume_key(claim),
            "user_id": claim.get("user_id", ""),
            "image_paths": claim.get("image_paths", ""),
            "claim_object": claim.get("claim_object", ""),
            "provider": self.client.provider,
            "model": self.client.model,
            "prompt_version": PROMPT_VERSION,
            "merge_policy_version": ADJUDICATION_MERGE_POLICY_VERSION,
            "reasons": reasons,
            "status": "attempted",
        }
        try:
            response = self.client.generate_json(
                prompt,
                images,
                cache_extra={
                    "pass": "adjudication",
                    "prompt_version": PROMPT_VERSION,
                    "claim": claim,
                    "user_history": user_history,
                    "evidence_requirements": requirements,
                    "preliminary_output": output_row,
                },
            )
            adjusted = apply_identity_conflict_policy(
                response,
                images,
                claim_object=claim.get("claim_object", ""),
            )
            adjusted = apply_wrong_object_taxonomy_policy(claim, adjusted)
            adjusted = apply_taxonomy_calibration(adjusted)
            adjusted = apply_package_missing_content_policy(claim, adjusted)
            adjusted = apply_user_history_flag_policy(adjusted, user_history)
            adjusted = apply_adjudication_merge_policy(
                preliminary=output_row,
                adjudicated=adjusted,
                images=images,
            )
            adjudicated = build_output_row(claim, adjusted, valid_image_ids=image_ids)
            changed_fields = [
                column
                for column, value in adjudicated.items()
                if output_row.get(column, "") != value
            ]
            record.update(
                {
                    "status": "succeeded",
                    "changed_fields": changed_fields,
                    "changed": bool(changed_fields),
                }
            )
            self.records.append(record)
            LOGGER.info("Adjudicated suspicious claim for user_id=%s", claim.get("user_id"))
            return adjudicated
        except (ModelUnavailable, ModelResponseError, ValueError) as exc:
            record.update({"status": "failed", "error": str(exc)})
            self.records.append(record)
            LOGGER.warning("Adjudication skipped after provider failure: %s", exc)
            return dict(output_row)


def _claim_resume_key(claim: Mapping[str, str]) -> str:
    return "\u241f".join(
        str(claim.get(column, "")).strip()
        for column in ("user_id", "image_paths", "user_claim", "claim_object")
    )
