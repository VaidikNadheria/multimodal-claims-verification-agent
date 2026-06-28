"""Terminal entry point for the HackerRank Orchestrate solution."""

from __future__ import annotations

import logging
import sys

from config import RuntimeConfig, build_arg_parser
from pipeline import ClaimPipeline
from validator import OutputValidationError


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)

    try:
        config = RuntimeConfig.from_args(args)
        pipeline = ClaimPipeline(config)
        pipeline.run()
    except (FileNotFoundError, ValueError, OutputValidationError) as exc:
        logging.getLogger(__name__).error("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
