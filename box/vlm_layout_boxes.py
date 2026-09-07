#!/usr/bin/env python3
"""Generate ordered foreground boxes with a VLM or an offline demo.

The VLM mode talks to any OpenAI-compatible chat-completions endpoint.  The
script deliberately keeps the response contract small and validates it before
the boxes are passed to the diffusion sampler.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib import request


def _image_size(path: Path | None, width: int, height: int) -> tuple[int, int]:
    if path is not None:
        try:
            from PIL import Image  # type: ignore

            with Image.open(path) as image:
                return int(image.width), int(image.height)
        except Exception:
            pass
    return width, height


def build_prompt(subjects: list[str], width: int, height: int) -> str:
    numbered = "\n".join(f"{i + 1}. {name}" for i, name in enumerate(subjects))
    return f"""You are a precise visual layout annotator. Inspect the supplied image.
Return exactly one tight, full-subject bounding box for every foreground entity,
not a part of the entity and not the background. The image coordinate system is
pixel coordinates with origin at the top-left; width={width}, height={height}.
Keep the entity order exactly as listed and return JSON only, with this schema:
{{"foreground_boxes":[{{"entity":"name","box":[x1,y1,x2,y2],"confidence":0.0}}]}}
Coordinates must satisfy 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}.
Entities to locate:
{numbered}
"""


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    return str(content)


def extract_json(text: str) -> dict[str, Any] | list[Any]:
    """Accept plain JSON, fenced JSON, or JSON surrounded by short prose."""
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, (dict, list)):
                return value
        except json.JSONDecodeError:
            pass
        for marker in ("{", "["):
            start = candidate.find(marker)
            if start >= 0:
                try:
                    value, _ = decoder.raw_decode(candidate[start:])
                    if isinstance(value, (dict, list)):
                        return value
                except json.JSONDecodeError:
                    continue
    raise ValueError("VLM response does not contain a JSON object or list")


def _box_values(value: Any) -> list[float]:
    if isinstance(value, dict):
        value = [value.get(k) for k in ("x1", "y1", "x2", "y2")]
    if isinstance(value, str):
        value = [x for x in re.split(r"[,\s]+", value.strip()) if x]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"invalid box: {value!r}")
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"box contains a non-numeric coordinate: {value!r}") from exc


def normalize_boxes(
    raw: dict[str, Any] | list[Any],
    subjects: list[str],
    width: int,
    height: int,
    margin: float = 0.04,
) -> dict[str, Any]:
    if isinstance(raw, dict):
        items = raw.get("foreground_boxes", raw.get("boxes"))
    else:
        items = raw
    if not isinstance(items, list):
        raise ValueError("JSON must contain a foreground_boxes list")

    by_name: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"each box must be an object, got {item!r}")
        name = str(item.get("entity", item.get("name", ""))).strip()
        if not name:
            raise ValueError("every box needs an entity/name")
        key = name.casefold()
        if key in by_name:
            raise ValueError(f"duplicate entity in VLM response: {name}")
        by_name[key] = item

    expected = [name.casefold() for name in subjects]
    missing = [name for name, key in zip(subjects, expected) if key not in by_name]
    extra = [name for key, item in by_name.items() if key not in expected for name in [str(item.get("entity", item.get("name", "")))]]
    if missing or extra or len(items) != len(subjects):
        raise ValueError(f"box/entity mismatch; missing={missing}, extra={extra}, count={len(items)}")

    normalized: list[dict[str, Any]] = []
    for subject in subjects:
        item = by_name[subject.casefold()]
        values = _box_values(item.get("box", item))
        x1, y1, x2, y2 = values
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        pad_x = max(0.0, x2 - x1) * margin
        pad_y = max(0.0, y2 - y1) * margin
        x1 = max(0.0, min(float(width), x1 - pad_x))
        y1 = max(0.0, min(float(height), y1 - pad_y))
        x2 = max(0.0, min(float(width), x2 + pad_x))
        y2 = max(0.0, min(float(height), y2 + pad_y))
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f"degenerate box for {subject}: {values}")
        confidence = item.get("confidence")
        try:
            confidence = None if confidence is None else max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = None
        result: dict[str, Any] = {
            "entity": subject,
            "box": [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))],
        }
        if confidence is not None:
            result["confidence"] = round(confidence, 4)
        normalized.append(result)
    return {"foreground_boxes": normalized, "image_size": [width, height], "box_method": "vlm"}


def demo_boxes(subjects: list[str], width: int, height: int, margin: float = 0.0) -> dict[str, Any]:
    """Create deterministic side-by-side boxes for format/integration tests."""
    n = len(subjects)
    boxes: list[dict[str, Any]] = []
    if n == 1:
        raw = [[0.16 * width, 0.18 * height, 0.84 * width, 0.96 * height]]
    else:
        gap = 0.04 * width
        usable = width - gap * (n + 1)
        item_w = usable / n
        raw = [[gap + i * (item_w + gap), 0.20 * height, gap + i * (item_w + gap) + item_w, 0.96 * height] for i in range(n)]
    for subject, box in zip(subjects, raw):
        boxes.append({"entity": subject, "box": box, "confidence": 0.5})
    return normalize_boxes({"foreground_boxes": boxes}, subjects, width, height, margin=margin) | {"box_method": "demo"}


def call_vlm(
    image: Path,
    subjects: list[str],
    width: int,
    height: int,
    endpoint: str,
    model: str,
    api_key: str,
    margin: float,
) -> dict[str, Any]:
    encoded = base64.b64encode(image.read_bytes()).decode("ascii")
    mime_type = mimetypes.guess_type(image.name)[0] or "application/octet-stream"
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": build_prompt(subjects, width, height)},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
        ]}],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with request.urlopen(req, timeout=180) as response:  # nosec B310 - endpoint is user supplied
        body = json.loads(response.read().decode("utf-8"))
    try:
        text = _content_to_text(body["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected VLM API response: {body!r}") from exc
    return normalize_boxes(extract_json(text), subjects, width, height, margin=margin)


def external_string(payload: dict[str, Any]) -> str:
    return "+".join(",".join(str(v) for v in item["box"]) for item in payload["foreground_boxes"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="preview image for --run")
    parser.add_argument("--subjects", nargs="+", default=["cat", "dog"])
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--margin", type=float, default=0.04)
    parser.add_argument("--endpoint", default=os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "local-vlm"))
    parser.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", ""))
    parser.add_argument("--run", action="store_true", help="call the VLM endpoint")
    parser.add_argument("--demo", action="store_true", help="use deterministic offline boxes")
    parser.add_argument("--format", choices=("json", "external"), default="json")
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run and args.demo:
        raise SystemExit("choose only one of --run and --demo")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if not 0.0 <= args.margin <= 0.5:
        raise SystemExit("--margin must be between 0 and 0.5")
    if len({name.casefold() for name in args.subjects}) != len(args.subjects):
        raise SystemExit("--subjects must not contain duplicate names")
    image = args.image
    width, height = _image_size(image, args.width, args.height)
    if args.run:
        if image is None or not image.is_file():
            raise SystemExit("--run requires an existing --image")
        payload = call_vlm(
            image,
            args.subjects,
            width,
            height,
            args.endpoint,
            args.model,
            args.api_key,
            args.margin,
        )
        payload["box_method"] = "vlm"
        payload["vlm_model"] = args.model
        payload["vlm_endpoint"] = args.endpoint
    else:
        payload = demo_boxes(args.subjects, width, height, margin=args.margin)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.format == "external":
        print(external_string(payload))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
