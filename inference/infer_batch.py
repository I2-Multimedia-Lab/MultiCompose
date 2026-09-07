#!/usr/bin/env python3
"""Run two-subject inference cases from a public TSV manifest."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent
INFER_ONE = PROJECT / "inference" / "infer_one_pair.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", type=Path, default=PROJECT / "runs" / "inference")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--resampling-steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=0.8)
    parser.add_argument("--t-cond", type=float, default=0.3)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    required = {"case_id", "theme1", "theme2", "background", "attr1", "attr2", "action"}
    if not rows:
        raise SystemExit(f"empty inference config: {path}")
    missing = required.difference(rows[0])
    if missing:
        raise SystemExit("inference config missing columns: " + ", ".join(sorted(missing)))
    return rows


def command_for(row: dict[str, str], args: argparse.Namespace) -> list[str]:
    case_id = row["case_id"].strip()
    if not case_id:
        raise ValueError("case_id cannot be empty")
    command = [
        sys.executable,
        str(INFER_ONE),
        "--theme1", row["theme1"].strip(),
        "--theme2", row["theme2"].strip(),
        "--background", row["background"].strip(),
        "--attr1", row["attr1"].strip(),
        "--attr2", row["attr2"].strip(),
        "--action", row["action"].strip(),
        "--mapping", str(args.mapping),
        "--model", args.model,
        "--output-dir", str(args.output_root / case_id),
        "--steps", str(args.steps),
        "--resampling-steps", str(args.resampling_steps),
        "--guidance", str(args.guidance),
        "--t-cond", str(args.t_cond),
        "--seed", (row.get("seed") or "1234").strip(),
        "--external-boxes", (row.get("external_boxes") or "120,220,500,960+540,220,920,960").strip(),
    ]
    if args.run:
        command.append("--run")
    return command


def main() -> int:
    args = parse_args()
    rows = read_cases(args.cases)
    failures = 0
    for index, row in enumerate(rows, start=1):
        command = command_for(row, args)
        print(f"[infer-batch] {index}/{len(rows)} {row['case_id']}")
        print(shlex.join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode:
            failures += 1
            if not args.continue_on_error:
                return completed.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
