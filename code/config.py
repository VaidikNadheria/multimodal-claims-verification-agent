"""Runtime configuration and CLI argument handling."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class RuntimeConfig:
    repo_root: Path
    dataset_dir: Path
    input_path: Path
    output_path: Path
    cache_dir: Path
    primary_provider: str
    fallback_provider: str
    model: Optional[str]
    limit: Optional[int]
    resume: bool
    dry_run: bool
    adjudicate_suspicious: bool
    no_adjudication: bool
    gemini_api_key: Optional[str]
    openai_api_key: Optional[str]
    gemini_model: str
    openai_model: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RuntimeConfig":
        repo_root = Path(__file__).resolve().parents[1]
        dataset_dir = _resolve_path(repo_root, args.dataset_dir)
        return cls(
            repo_root=repo_root,
            dataset_dir=dataset_dir,
            input_path=_resolve_path(repo_root, args.input),
            output_path=_resolve_path(repo_root, args.output),
            cache_dir=repo_root / ".cache" / "vlm",
            primary_provider=args.primary_provider.lower(),
            fallback_provider=args.fallback_provider.lower(),
            model=args.model,
            limit=args.limit,
            resume=args.resume,
            dry_run=args.dry_run,
            adjudicate_suspicious=args.adjudicate_suspicious,
            no_adjudication=args.no_adjudication,
            gemini_api_key=os.environ.get("GEMINI_API_KEY"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            gemini_model=os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
            openai_model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        )

    def model_for_provider(self, provider: str) -> str:
        provider = provider.lower()
        if self.model:
            return self.model
        if provider == "gemini":
            return self.gemini_model
        if provider == "openai":
            return self.openai_model
        return self.model or "unknown"

    def api_key_for_provider(self, provider: str) -> Optional[str]:
        provider = provider.lower()
        if provider == "gemini":
            return self.gemini_api_key
        if provider == "openai":
            return self.openai_api_key
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the multimodal evidence review pipeline."
    )
    parser.add_argument("--input", default="dataset/claims.csv")
    parser.add_argument("--output", default="output.csv")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--primary-provider", default="gemini")
    parser.add_argument("--fallback-provider", default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--adjudicate-suspicious", action="store_true")
    parser.add_argument("--no-adjudication", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser


def _resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repo_root / path

