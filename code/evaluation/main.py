"""Evaluation entry point skeleton for sample_claims.csv.

This file intentionally performs no real model calls by default. It can be
expanded to compare prompts/providers and write evaluation/evaluation_report.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from config import RuntimeConfig, build_arg_parser  # noqa: E402
from pipeline import ClaimPipeline  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a dry-run evaluation scaffold on sample_claims.csv."
    )
    parser.add_argument("--output", default="code/evaluation/sample_dry_run_output.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pipeline_parser = build_arg_parser()
    pipeline_args = pipeline_parser.parse_args(
        [
            "--input",
            "dataset/sample_claims.csv",
            "--output",
            args.output,
            "--dry-run",
            "--no-adjudication",
            *([] if args.limit is None else ["--limit", str(args.limit)]),
        ]
    )
    config = RuntimeConfig.from_args(pipeline_args)
    ClaimPipeline(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
