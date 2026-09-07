#!/usr/bin/env python3
"""Run concept-personalization training jobs from a public TSV manifest."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
TRAIN_ONE = PROJECT / "training" / "train_one_concept.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT / "runs" / "train")
    parser.add_argument("--max-train-steps", type=int, default=1500)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"concept_id", "modifier_token", "instance_prompt", "dataset_dir"}
    if not rows:
        raise SystemExit(f"empty training config: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise SystemExit("training config missing columns: " + ", ".join(sorted(missing)))
    return rows


def command_for(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    concept_id = row["concept_id"].strip()
    if not concept_id:
        raise ValueError("concept_id cannot be empty")
    command = [
        sys.executable,
        str(TRAIN_ONE),
        "--concept-id", concept_id,
        "--dataset-dir", row["dataset_dir"].strip(),
        "--instance-prompt", row["instance_prompt"].strip(),
        "--modifier-token", row["modifier_token"].strip(),
        "--model", args.model,
        "--output-dir", str(args.output_root / concept_id),
        "--max-train-steps", str(args.max_train_steps),
        "--save-steps", str(args.save_steps),
        "--resolution", str(args.resolution),
        "--learning-rate", str(args.learning_rate),
    ]
    if args.run:
        command.append("--run")
    return command


def main() -> int:
    args = parse_args()
    rows = read_manifest(args.config)
    failures = 0
    for index, row in enumerate(rows, start=1):
        command = command_for(row, args)
        print(f"[train-batch] {index}/{len(rows)} {row['concept_id']}")
        print(shlex.join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failures += 1
            if not args.continue_on_error:
                return completed.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
