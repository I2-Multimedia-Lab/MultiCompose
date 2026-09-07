#!/usr/bin/env python3
"""Traditional CLIP evaluation for two personalized subjects plus background.

It reports subject identity, global prompt alignment, correct attribute binding,
swapped-attribute confusion, and background alignment. Unlike the VLM rubric,
this output is a 0..1 similarity score for any compatible two-subject sample.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageFile

IMPORT_ERROR: Exception | None = None
try:
    import torch
    from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor
except Exception as exc:  # pragma: no cover - depends on runtime environment
    torch = None  # type: ignore[assignment]
    CLIPModel = None  # type: ignore[assignment]
    CLIPProcessor = None  # type: ignore[assignment]
    AutoImageProcessor = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    IMPORT_ERROR = exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ImageFile.LOAD_TRUNCATED_IMAGES = True


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def to_unit(cosine: float) -> float:
    return max(0.0, min(1.0, 0.5 * (cosine + 1.0)))


def image_cosine_01(value: float) -> float:
    """DINO/CLIP image cosine is conventionally reported directly."""
    return max(0.0, min(1.0, value))


def cosine(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
    a = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    b = b / b.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return a @ b.T


def parse_boxes(value: str, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for raw_box in value.split("+"):
        try:
            values = [int(round(float(x))) for x in raw_box.split(",")]
        except ValueError as exc:
            raise ValueError(f"invalid box: {raw_box!r}") from exc
        if len(values) != 4:
            raise ValueError(f"box needs four coordinates: {raw_box!r}")
        x1, y1, x2, y2 = values
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"degenerate box after clipping: {raw_box!r}")
        boxes.append((x1, y1, x2, y2))
    if len(boxes) != 2:
        raise ValueError("--boxes must contain exactly two boxes separated by '+'")
    return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated-image", type=Path, default=Path("/path/to/generated.png"))
    parser.add_argument("--subject-a-ref-dir", type=Path, default=Path("/path/to/subject_a"))
    parser.add_argument("--subject-b-ref-dir", type=Path, default=Path("/path/to/subject_b"))
    parser.add_argument("--background-ref-dir", type=Path, default=Path("/path/to/background"))
    parser.add_argument("--subject-a", default="subject_a")
    parser.add_argument("--subject-b", default="subject_b")
    parser.add_argument("--attr-a", default="attribute_one")
    parser.add_argument("--attr-b", default="attribute_two")
    parser.add_argument("--background", default="background_scene")
    parser.add_argument("--prompt", default="photo of subject_a and subject_b in background_scene")
    parser.add_argument("--boxes", default="", help="ordered A+B boxes: x1,y1,x2,y2+x1,y1,x2,y2")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--identity-model", default="facebook/dinov2-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--topk-refs", type=int, default=3)
    parser.add_argument("--identity-weight", type=float, default=0.6)
    parser.add_argument("--binding-weight", type=float, default=0.4)
    parser.add_argument("--confusion-lambda", type=float, default=0.7)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--run", action="store_true", help="load CLIP and score; default is dry-run")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run:
        print("[traditional-eval] dry-run; no model was loaded")
        print(json.dumps({
            "generated_image": str(args.generated_image),
            "subject_a": args.subject_a,
            "subject_b": args.subject_b,
            "correct_binding_queries": [
                f"a {args.subject_a} wearing {args.attr_a}",
                f"a {args.subject_b} wearing {args.attr_b}",
            ],
            "swapped_binding_queries": [
                f"a {args.subject_a} wearing {args.attr_b}",
                f"a {args.subject_b} wearing {args.attr_a}",
            ],
            "identity_weight": args.identity_weight,
            "binding_weight": args.binding_weight,
            "confusion_lambda": args.confusion_lambda,
        }, ensure_ascii=False, indent=2))
        return 0

    if torch is None or CLIPModel is None or CLIPProcessor is None or AutoImageProcessor is None or AutoModel is None:
        raise RuntimeError(f"torch/transformers unavailable: {IMPORT_ERROR}")
    if abs(args.identity_weight + args.binding_weight - 1.0) > 1e-6:
        raise SystemExit("identity-weight + binding-weight must equal 1")
    required = [args.generated_image, args.subject_a_ref_dir, args.subject_b_ref_dir, args.background_ref_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing paths: " + ", ".join(missing))

    refs = [list_images(path) for path in required[1:]]
    if any(not paths for paths in refs):
        raise SystemExit("each reference directory must contain at least one image")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model = CLIPModel.from_pretrained(args.clip_model, use_safetensors=True, local_files_only=args.local_files_only).to(device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model, local_files_only=args.local_files_only)
    identity_model = AutoModel.from_pretrained(args.identity_model, local_files_only=args.local_files_only).to(device)
    identity_processor = AutoImageProcessor.from_pretrained(args.identity_model, local_files_only=args.local_files_only)
    clip_model.eval()
    identity_model.eval()

    @torch.inference_mode()
    def clip_image_features(images: list[Image.Image]) -> "torch.Tensor":
        values = clip_processor(images=images, return_tensors="pt")
        return clip_model.get_image_features(**{key: value.to(device) for key, value in values.items()}).detach().cpu()

    @torch.inference_mode()
    def text_features(texts: list[str]) -> "torch.Tensor":
        values = clip_processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        return clip_model.get_text_features(**{key: value.to(device) for key, value in values.items()}).detach().cpu()

    @torch.inference_mode()
    def identity_features(images: list[Image.Image]) -> "torch.Tensor":
        values = identity_processor(images=images, return_tensors="pt")
        output = identity_model(**{key: value.to(device) for key, value in values.items()})
        features = getattr(output, "pooler_output", None)
        if features is None:
            features = output.last_hidden_state[:, 0]
        return features.detach().cpu()

    generated_image = open_rgb(args.generated_image)
    if args.boxes:
        boxes = parse_boxes(args.boxes, generated_image.width, generated_image.height)
        subject_images = [generated_image.crop(box) for box in boxes]
        crop_warning = ""
    else:
        boxes = []
        subject_images = [generated_image, generated_image]
        crop_warning = "No boxes supplied: both subject metrics use the full image and are less reliable."
    generated_identity_images = [subject_images[0], subject_images[1], generated_image]
    generated_identity = identity_features(generated_identity_images)
    ref_features = [identity_features([open_rgb(path) for path in paths]) for paths in refs]
    identities: list[float] = []
    for index, features in enumerate(ref_features):
        similarities = cosine(generated_identity[index : index + 1], features).squeeze(0)
        k = min(args.topk_refs, similarities.numel())
        identities.append(image_cosine_01(float(similarities.topk(k).values.mean())))
    identity_min = min(identities)
    identity_avg = sum(identities) / len(identities)
    identity_score = 0.7 * identity_min + 0.3 * identity_avg

    queries = [
        args.prompt,
        f"a {args.subject_a} wearing {args.attr_a}",
        f"a {args.subject_b} wearing {args.attr_b}",
        f"a {args.subject_a} wearing {args.attr_b}",
        f"a {args.subject_b} wearing {args.attr_a}",
        f"{args.background} background",
    ]
    query_features = text_features(queries)
    full_clip = clip_image_features([generated_image])
    crop_clip = clip_image_features(subject_images)
    global_prompt = to_unit(float(cosine(full_clip, query_features[0:1]).item()))
    correct_a = to_unit(float(cosine(crop_clip[0:1], query_features[1:2]).item()))
    correct_b = to_unit(float(cosine(crop_clip[1:2], query_features[2:3]).item()))
    swapped_a = to_unit(float(cosine(crop_clip[0:1], query_features[3:4]).item()))
    swapped_b = to_unit(float(cosine(crop_clip[1:2], query_features[4:5]).item()))
    background_text = to_unit(float(cosine(full_clip, query_features[5:6]).item()))
    confusion = 0.5 * (max(0.0, swapped_a - correct_a) + max(0.0, swapped_b - correct_b))
    binding_raw = 0.25 * (correct_a + correct_b + global_prompt + background_text)
    binding_score = max(0.0, min(1.0, binding_raw - args.confusion_lambda * confusion))
    total = args.identity_weight * identity_score + args.binding_weight * binding_score

    result = {
        "method": "traditional_dino_clip_pair",
        "identity_model": args.identity_model,
        "binding_model": args.clip_model,
        "device": device,
        "generated_image": str(args.generated_image),
        "subjects": [args.subject_a, args.subject_b],
        "attributes": [args.attr_a, args.attr_b],
        "background": args.background,
        "boxes": [list(box) for box in boxes],
        "crop_warning": crop_warning,
        "identity": {
            "subject_a": round(identities[0], 6),
            "subject_b": round(identities[1], 6),
            "background": round(identities[2], 6),
            "minimum": round(identity_min, 6),
            "average": round(identity_avg, 6),
            "score": round(identity_score, 6),
        },
        "binding": {
            "global_prompt": round(global_prompt, 6),
            "correct_a": round(correct_a, 6),
            "correct_b": round(correct_b, 6),
            "swapped_a": round(swapped_a, 6),
            "swapped_b": round(swapped_b, 6),
            "confusion_penalty": round(confusion, 6),
            "score": round(binding_score, 6),
        },
        "weights": {
            "identity": args.identity_weight,
            "binding": args.binding_weight,
            "confusion_lambda": args.confusion_lambda,
        },
        "final_score": round(total, 6),
    }
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(output, end="")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
