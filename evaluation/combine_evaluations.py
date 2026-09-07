#!/usr/bin/env python3
"""Combine VLM and traditional pair-evaluation JSON without hiding either score."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def nested(data: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    value: Any = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-json", type=Path, required=True)
    parser.add_argument("--traditional-json", type=Path, required=True)
    parser.add_argument("--vlm-weight", type=float, default=0.5)
    parser.add_argument("--traditional-weight", type=float, default=0.5)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if abs(args.vlm_weight + args.traditional_weight - 1.0) > 1e-6:
        raise SystemExit("vlm-weight + traditional-weight must equal 1")
    vlm = json.loads(args.vlm_json.read_text(encoding="utf-8"))
    traditional = json.loads(args.traditional_json.read_text(encoding="utf-8"))
    vlm_score_100 = nested(vlm, "score")
    traditional_score_01 = nested(traditional, "final_score")
    traditional_score_100 = 100.0 * traditional_score_01
    combined = args.vlm_weight * vlm_score_100 + args.traditional_weight * traditional_score_100
    result = {
        "synthetic_example": bool(vlm.get("synthetic_example") or traditional.get("synthetic_example")),
        "method": "vlm_plus_traditional_pair",
        "vlm_score_100": round(vlm_score_100, 4),
        "traditional_score_01": round(traditional_score_01, 6),
        "traditional_score_100": round(traditional_score_100, 4),
        "weights": {"vlm": args.vlm_weight, "traditional": args.traditional_weight},
        "combined_score_100": round(combined, 4),
        "components": {
            "vlm_deductions": vlm.get("scores_breakdown", {}),
            "traditional_identity": traditional.get("identity", {}),
            "traditional_binding": traditional.get("binding", {}),
        },
        "warning": "Combined score is optional; always report the two component methods as well.",
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(output, end="")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
