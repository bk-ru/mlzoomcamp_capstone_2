"""CLI entrypoint for model training."""

from __future__ import annotations

import argparse

from ..config import DEFAULT_MAX_ROWS, DEFAULT_PAGE_SIZE
from .pipeline import run_training


def main() -> None:
    """Run model training and save the best model artifacts."""
    parser = argparse.ArgumentParser(description="Train overtime prediction models")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    args = parser.parse_args()
    run_training(max_rows=args.max_rows, page_size=args.page_size)


if __name__ == "__main__":
    main()
