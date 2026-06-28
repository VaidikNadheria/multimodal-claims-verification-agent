"""Gemini vision client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Sequence

from image_utils import ImageEvidence, encode_image_for_model
from prompts import SYSTEM_INSTRUCTIONS
from vlm_client import ModelResponseError, TransientModelError, VisionClient


class GeminiVisionClient(VisionClient):
    provider = "gemini"

    def _call_model(self, prompt: str, images: Sequence[ImageEvidence]) -> str:
        parts = [{"text": f"{SYSTEM_INSTRUCTIONS}\n\n{prompt}"}]
        for image in images:
            if not image.exists:
                continue
            encoded = encode_image_for_model(image.absolute_path)
            parts.append(
                {
                    "inline_data": {
                        "mime_type": encoded.mime_type,
                        "data": encoded.base64_data,
                    }
                }
            )

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key or "",
            },
            method="POST",
        )
        try:
            with self._urlopen(request) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in {408, 409, 429, 500, 502, 503, 504}:
                raise TransientModelError(f"Gemini transient HTTP status {exc.code}") from exc
            raise ModelResponseError(f"Gemini HTTP status {exc.code}") from exc
        except urllib.error.URLError as exc:
            raise TransientModelError(f"Gemini request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TransientModelError("Gemini request timed out") from exc
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"Gemini request failed: {exc}") from exc

        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError(f"Unexpected Gemini response shape: {data}") from exc
