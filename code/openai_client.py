"""OpenAI-compatible vision client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Sequence

from image_utils import ImageEvidence, encode_image_for_openai
from prompts import SYSTEM_INSTRUCTIONS
from vlm_client import ModelResponseError, TransientModelError, VisionClient


class OpenAIVisionClient(VisionClient):
    provider = "openai"

    def _call_model(self, prompt: str, images: Sequence[ImageEvidence]) -> str:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            if not image.exists:
                continue
            try:
                encoded = encode_image_for_openai(image.absolute_path)
            except RuntimeError as exc:
                raise ModelResponseError(
                    f"Could not prepare image for OpenAI vision: {exc}"
                ) from exc
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{encoded.mime_type};base64,{encoded.base64_data}",
                    },
                }
            )
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": content},
            ],
        }
        if not self.model.lower().startswith("gpt-5"):
            payload["temperature"] = 0
        request = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with self._urlopen(request) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            message = _safe_error_message(body)
            if exc.code in {408, 409, 429, 500, 502, 503, 504}:
                raise TransientModelError(
                    f"OpenAI transient HTTP status {exc.code}: {message}"
                ) from exc
            raise ModelResponseError(f"OpenAI HTTP status {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise TransientModelError(f"OpenAI request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise TransientModelError("OpenAI request timed out") from exc
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"OpenAI request failed: {exc}") from exc

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelResponseError(f"Unexpected OpenAI response shape: {data}") from exc


def _safe_error_message(body: str) -> str:
    try:
        data = json.loads(body)
        message = data.get("error", {}).get("message", body)
    except json.JSONDecodeError:
        message = body
    return " ".join(str(message).split())[:500]
