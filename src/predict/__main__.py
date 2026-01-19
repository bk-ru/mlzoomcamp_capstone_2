"""CLI for local predictions using a saved model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .predictor import load_artifacts, predict_with_model


def main() -> None:
    """Run a prediction from a JSON file."""
    parser = argparse.ArgumentParser(description="Run local predictions")
    parser.add_argument("--input", required=True, help="Path to JSON input file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text())
    if isinstance(payload, dict):
        records = [payload]
        single = True
    elif isinstance(payload, list):
        records = payload
        single = False
    else:
        raise ValueError("Input JSON must be an object or a list of objects")

    model, meta = load_artifacts()
    outputs = predict_with_model(records, model, meta)

    print(json.dumps(outputs[0] if single else outputs, indent=2))


if __name__ == "__main__":
    main()
