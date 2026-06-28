"""Provider-independent vision model client primitives with JSON caching."""

from __future__ import annotations

import hashlib
import ssl
import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence
import urllib.request

from image_utils import ImageEvidence

LOGGER = logging.getLogger(__name__)


class ModelUnavailable(RuntimeError):
    """Raised when a provider cannot be called because configuration is missing."""


class ModelResponseError(RuntimeError):
    """Raised when a provider response is not parseable or usable."""


class TransientModelError(ModelResponseError):
    """Raised when a provider failure is likely retryable."""


class VisionClient:
    provider = "base"

    def __init__(
        self,
        model: str,
        api_key: str | None,
        cache_dir: Path,
        timeout_seconds: int = 90,
        max_retries: int = 3,
        retry_base_seconds: float = 1.0,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.cache_dir = cache_dir / self.provider / self.model.replace("/", "_")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def generate_json(
        self,
        prompt: str,
        images: Sequence[ImageEvidence],
        cache_extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.available:
            raise ModelUnavailable(
                f"{self.provider} provider is unavailable because its API key is not set"
            )

        cache_path = self._cache_path(prompt, images, cache_extra or {})
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            LOGGER.info("Using cached %s response: %s", self.provider, cache_path)
            return cached["parsed_json"]

        started = time.monotonic()
        raw_text = self._call_model_with_retries(prompt, images)
        parsed = parse_json_object(raw_text)
        elapsed = time.monotonic() - started

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "provider": self.provider,
                    "model": self.model,
                    "elapsed_seconds": round(elapsed, 3),
                    "parsed_json": parsed,
                    "raw_text": raw_text,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        LOGGER.info("Cached %s response: %s", self.provider, cache_path)
        return parsed

    def _call_model(self, prompt: str, images: Sequence[ImageEvidence]) -> str:
        raise NotImplementedError

    def _call_model_with_retries(
        self,
        prompt: str,
        images: Sequence[ImageEvidence],
    ) -> str:
        last_error: TransientModelError | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self._call_model(prompt, images)
            except TransientModelError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self.retry_base_seconds * (2 ** (attempt - 1))
                delay += random.uniform(0, 0.25)
                LOGGER.warning(
                    "%s transient API failure on attempt %s/%s; retrying in %.2fs: %s",
                    self.provider,
                    attempt,
                    self.max_retries,
                    delay,
                    exc,
                )
                time.sleep(delay)
        raise last_error or TransientModelError(f"{self.provider} transient request failed")

    def _cache_path(
        self,
        prompt: str,
        images: Sequence[ImageEvidence],
        cache_extra: Mapping[str, Any],
    ) -> Path:
        digest = hashlib.sha256()
        digest.update(self.provider.encode("utf-8"))
        digest.update(self.model.encode("utf-8"))
        digest.update(prompt.encode("utf-8"))
        digest.update(json.dumps(cache_extra, sort_keys=True).encode("utf-8"))
        for image in images:
            digest.update(image.csv_path.encode("utf-8"))
            digest.update(image.sha256.encode("utf-8"))
        return self.cache_dir / f"{digest.hexdigest()}.json"

    def _urlopen(self, request: urllib.request.Request):
        """Open HTTPS requests with a reliable CA bundle when available."""

        context = None
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            context = ssl.create_default_context()
        return urllib.request.urlopen(
            request,
            timeout=self.timeout_seconds,
            context=context,
        )


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = _strip_markdown_fence(text.strip())
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ModelResponseError("model response did not contain a JSON object")
        candidate = _remove_trailing_commas(stripped[start : end + 1])
        value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ModelResponseError("model response JSON was not an object")
    return value


def _strip_markdown_fence(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)
