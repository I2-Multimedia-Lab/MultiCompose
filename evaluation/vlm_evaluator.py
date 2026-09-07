#!/usr/bin/env python3
"""Score one personalized multi-subject image with an OpenAI-compatible VLM.

The evaluator uses a structured 100-point deduction rubric. The default is a
no-network dry run.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any

import requests
from PIL import Image


def build_rubric(subject_a: str, subject_b: str, background: str, prompt: str) -> str:
    return f"""# 多概念个性化生成定向评估（100分扣分制）
请比较生成图、主体A参考图、主体B参考图和背景参考图。
主体A：{subject_a}
主体B：{subject_b}
背景：{background}
提示词：{prompt}

从100分开始扣分：
1. 概念召回：漏掉主体A或主体B各扣20分；缺背景扣10分；缺一个关键属性扣10分；两个主体都不存在时总分为0。召回扣分字段限制在0到-50。
2. 特征解耦与属性绑定：严重跨主体特征融合扣15分；主体A和主体B的属性张冠李戴扣15分。绑定扣分字段限制在0到-30。
3. 身份保真：主体A不像参考图扣10分；主体B不像参考图扣10分；背景严重偏离参考图扣10分。保真扣分字段限制在0到-30。

只输出JSON：
{{
  "reasoning": "一句话主要理由",
  "missing_entities": [],
  "scores_breakdown": {{
    "recall_deduction": 0,
    "binding_deduction": 0,
    "fidelity_deduction": 0
  }},
  "score": 100
}}
"""


def encode_image(path: Path, thumbnail: bool) -> str:
    if thumbnail:
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((768, 768))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
        mime, payload = "image/jpeg", buffer.getvalue()
    else:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        payload = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def extract_json(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.I | re.S))
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = candidate.find("{")
        if start >= 0:
            try:
                value, _ = decoder.raw_decode(candidate[start:])
                if isinstance(value, dict):
                    return value
            except json.JSONDecodeError:
                pass
    raise ValueError("VLM response contains no JSON object")


def deduction(value: Any, floor: float) -> float:
    if isinstance(value, (int, float)):
        parsed = float(value)
    else:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
        parsed = float(match.group()) if match else 0.0
    if parsed > 0:
        parsed = -parsed
    return max(floor, min(0.0, parsed))


def normalize_result(raw: dict[str, Any]) -> dict[str, Any]:
    breakdown = raw.get("scores_breakdown")
    if not isinstance(breakdown, dict):
        breakdown = {}
    recall = deduction(breakdown.get("recall_deduction"), -50.0)
    binding = deduction(breakdown.get("binding_deduction"), -30.0)
    fidelity = deduction(breakdown.get("fidelity_deduction"), -30.0)
    derived = max(0.0, min(100.0, 100.0 + recall + binding + fidelity))
    try:
        reported = max(0.0, min(100.0, float(raw.get("score", derived))))
    except (TypeError, ValueError):
        reported = derived
    missing = raw.get("missing_entities", [])
    if not isinstance(missing, list):
        missing = [str(missing)]
    return {
        "score": round(reported, 2),
        "derived_score": round(derived, 2),
        "score_consistent": abs(reported - derived) <= 1.0,
        "scores_breakdown": {
            "recall_deduction": round(recall, 2),
            "binding_deduction": round(binding, 2),
            "fidelity_deduction": round(fidelity, 2),
        },
        "missing_entities": [str(x) for x in missing],
        "reasoning": str(raw.get("reasoning", "")),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-image", type=Path, default=Path("/path/to/generated.png"))
    parser.add_argument("--subject-a-ref", type=Path, default=Path("/path/to/subject_a/0.jpg"))
    parser.add_argument("--subject-b-ref", type=Path, default=Path("/path/to/subject_b/0.jpg"))
    parser.add_argument("--background-ref", type=Path, default=Path("/path/to/background/0.jpg"))
    parser.add_argument("--subject-a", default="subject_a")
    parser.add_argument("--subject-b", default="subject_b")
    parser.add_argument("--background", default="background_scene")
    parser.add_argument("--prompt", default="photo of subject_a and subject_b in background_scene")
    parser.add_argument("--endpoint", default=os.environ.get("VLM_ENDPOINT", "http://127.0.0.1:8000/v1/chat/completions"))
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", "your-vlm-model"))
    parser.add_argument("--api-key", default=os.environ.get("VLM_API_KEY", os.environ.get("DASHSCOPE_API_KEY", "")))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--run", action="store_true", help="send the API request; default only prints the rubric")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rubric = build_rubric(args.subject_a, args.subject_b, args.background, args.prompt)
    if not args.run:
        print("[vlm-eval] dry-run; no API request was sent")
        print(f"[vlm-eval] endpoint={args.endpoint} model={args.model}")
        print(f"[vlm-eval] generated={args.generated_image}")
        print(rubric)
        return 0

    paths = [args.subject_a_ref, args.subject_b_ref, args.background_ref, args.generated_image]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit("missing image files: " + ", ".join(missing))

    content = [{"type": "text", "text": rubric}]
    for path, thumbnail in zip(paths, (True, True, True, False)):
        content.append({"type": "image_url", "image_url": {"url": encode_image(path, thumbnail)}})
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    response = requests.post(
        args.endpoint,
        headers=headers,
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 800,
            "temperature": args.temperature,
        },
        timeout=args.timeout,
    )
    response.raise_for_status()
    body = response.json()
    try:
        response_text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"unexpected API response: {body!r}") from exc
    result = normalize_result(extract_json(str(response_text)))
    result.update({
        "generated_image": str(args.generated_image),
        "reference_images": [str(x) for x in paths[:3]],
        "subjects": [args.subject_a, args.subject_b],
        "background": args.background,
        "prompt": args.prompt,
        "model": args.model,
        "endpoint": args.endpoint,
    })
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(output, end="")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(output, encoding="utf-8")
    if not result["score_consistent"]:
        print("[vlm-eval] warning: model score and deductions disagree", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
