"""Image metadata, hashing, and base64 helpers."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable


MAX_MODEL_IMAGE_SIDE = 2048
MAX_MODEL_IMAGE_BYTES = 4 * 1024 * 1024
JPEG_QUALITY = 92
MODEL_DIRECT_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
# OpenAI accepts WebP, but normalizing it avoids format-specific handling and
# keeps every OpenAI image request to the two most broadly supported formats.
OPENAI_DIRECT_MIME_TYPES = frozenset({"image/jpeg", "image/png"})

_PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "AVIF": "image/avif",
    "HEIF": "image/heic",
    "HEIC": "image/heic",
}


@dataclass(frozen=True)
class ImageEvidence:
    image_id: str
    csv_path: str
    absolute_path: Path
    exists: bool
    sha256: str
    byte_size: int
    mime_type: str


@dataclass(frozen=True)
class EncodedImage:
    mime_type: str
    base64_data: str
    original_bytes: int
    encoded_bytes: int
    resized: bool


def parse_image_paths(image_paths: str) -> list[str]:
    return [part.strip() for part in image_paths.split(";") if part.strip()]


def image_id_from_path(image_path: str) -> str:
    return Path(image_path).stem


def collect_image_evidence(dataset_dir: Path, image_paths: str) -> list[ImageEvidence]:
    evidence = []
    for csv_path in parse_image_paths(image_paths):
        candidate_path = Path(csv_path)
        absolute_path = candidate_path if candidate_path.is_absolute() else dataset_dir / candidate_path
        exists = absolute_path.exists()
        evidence.append(
            ImageEvidence(
                image_id=image_id_from_path(csv_path),
                csv_path=csv_path,
                absolute_path=absolute_path,
                exists=exists,
                sha256=file_sha256(absolute_path) if exists else "",
                byte_size=absolute_path.stat().st_size if exists else 0,
                mime_type=mime_type_for_path(absolute_path),
            )
        )
    return evidence


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_image_base64(path: Path) -> str:
    return encode_image_for_model(path).base64_data


def encode_image_for_model(
    path: Path,
    direct_mime_types: Iterable[str] = MODEL_DIRECT_MIME_TYPES,
) -> EncodedImage:
    """Encode an image for VLM input, downscaling only when it is large.

    Damage details matter, so the default keeps originals unless an image is very
    large. When resizing is needed, the longest side is capped at 2048px and JPEG
    quality remains high.
    """

    raw = path.read_bytes()
    original_mime = mime_type_for_bytes(raw)
    direct_mime_types = set(direct_mime_types)
    if len(raw) <= MAX_MODEL_IMAGE_BYTES and original_mime in direct_mime_types:
        return EncodedImage(
            mime_type=original_mime,
            base64_data=base64.b64encode(raw).decode("ascii"),
            original_bytes=len(raw),
            encoded_bytes=len(raw),
            resized=False,
        )

    try:
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is required to normalize unsupported image formats for model input."
        ) from exc

    try:
        with Image.open(BytesIO(raw)) as image:
            image = ImageOps.exif_transpose(image)
            image.thumbnail((MAX_MODEL_IMAGE_SIDE, MAX_MODEL_IMAGE_SIDE))
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            encoded = _encode_jpeg_with_size_limit(image)
    except Exception as exc:
        raise RuntimeError(f"Could not normalize image {path.name} for model input") from exc

    return EncodedImage(
        mime_type="image/jpeg",
        base64_data=base64.b64encode(encoded).decode("ascii"),
        original_bytes=len(raw),
        encoded_bytes=len(encoded),
        resized=len(encoded) != len(raw),
    )


def encode_image_for_openai(path: Path) -> EncodedImage:
    """Return an OpenAI-safe JPEG/PNG image based on actual file contents.

    Dataset filenames are not trusted: an AVIF payload may be named ``.jpg``.
    JPEG and PNG bytes are preserved when already within size limits; every
    other format is decoded and re-encoded as a high-quality JPEG.
    """

    return encode_image_for_model(path, direct_mime_types=OPENAI_DIRECT_MIME_TYPES)


def mime_type_for_path(path: Path) -> str:
    return mime_type_for_bytes(path.read_bytes())


def mime_type_for_bytes(data: bytes) -> str:
    """Identify image MIME type by signature, then Pillow, never filename."""

    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12].lower()
        if brand in {b"avif", b"avis"}:
            return "image/avif"
        if brand in {b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1"}:
            return "image/heic"

    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            return _PIL_FORMAT_TO_MIME.get(image.format or "", "application/octet-stream")
    except Exception:
        return "application/octet-stream"


def _encode_jpeg_with_size_limit(image) -> bytes:
    """Preserve detail while bounding payload size for vision APIs."""

    working = image
    quality = JPEG_QUALITY
    for _ in range(4):
        output = BytesIO()
        working.save(output, format="JPEG", quality=quality, optimize=True)
        encoded = output.getvalue()
        if len(encoded) <= MAX_MODEL_IMAGE_BYTES:
            return encoded
        quality = max(70, quality - 8)
    return encoded

