#!/usr/bin/env python3
"""Build or run one two-subject plus background inference case.

The command mirrors the production shell options while keeping one case
explicit. It is dry-run by default.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_MAPPING = PROJECT / "configs" / "concepts.example.tsv"
DEFAULT_SAMPLER = PROJECT / "source" / "inference" / "multiconcept_sampler.py"


def load_mapping(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return {r["concept_id"]: r for r in csv.DictReader(f, delimiter="\t") if r.get("concept_id")}


def pick(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--theme1", default="subject_a")
    p.add_argument("--theme2", default="subject_b")
    p.add_argument("--background", default="background_scene")
    p.add_argument("--attr1", default="attribute_one")
    p.add_argument("--attr2", default="attribute_two")
    p.add_argument("--action", default="standing")
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--sampler", type=Path, default=DEFAULT_SAMPLER)
    p.add_argument("--model", default=os.environ.get("MODEL_NAME", "/path/to/models/stable-diffusion-xl-base-1.0"))
    p.add_argument("--output-dir", type=Path, default=PROJECT / "runs" / "inference_demo")
    p.add_argument("--guidance", type=float, default=0.80)
    p.add_argument("--t-cond", type=float, default=0.30)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--resampling-steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--external-boxes", default="120,300,520,980+600,200,1000,1000")
    p.add_argument("--run", action="store_true", help="actually run the sampler; default is dry-run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mapping = load_mapping(args.mapping)
    missing = [x for x in (args.theme1, args.theme2, args.background) if x not in mapping]
    if missing:
        raise SystemExit(f"concepts missing from mapping: {', '.join(missing)}")
    a, b, bg = (mapping[x] for x in (args.theme1, args.theme2, args.background))
    token_a = pick(a, "modifier_token", "personalized_token")
    token_b = pick(b, "modifier_token", "personalized_token")
    token_bg = pick(bg, "modifier_token", "personalized_token")
    class_a = pick(a, "class_word") or args.theme1
    class_b = pick(b, "class_word") or args.theme2
    class_bg = pick(bg, "class_word") or args.background
    label_a = class_a.split()[0]
    label_b = class_b.split()[0]
    label_bg = class_bg.split()[0]
    fg_a = pick(a, "final_classdiffusion_weight_path", "final_personalized_weight_path", "multicompose_final_weight_path")
    fg_b = pick(b, "final_classdiffusion_weight_path", "final_personalized_weight_path", "multicompose_final_weight_path")
    bg_ckpt = pick(bg, "multicompose_final_weight_path", "final_personalized_weight_path", "final_classdiffusion_weight_path")
    prompt = (
        f"photo of a {token_a} {class_a} wearing {args.attr1} and a "
        f"{token_b} {class_b} wearing {args.attr2}, {args.action}, {token_bg} {class_bg} background"
    )
    clean = f"photo of a {class_a} wearing {args.attr1} and a {class_b} wearing {args.attr2}, {args.action}, {class_bg} background"
    out = args.output_dir.resolve()
    checkpoint = "+".join((fg_a, fg_b, bg_ckpt))
    cmd = [
        "python3", str(args.sampler.resolve()),
        "--guidance_scale", str(args.guidance), "--n_timesteps", str(args.steps),
        "--prompt", prompt, "--pretrained_model_name_or_path", args.model,
        "--personal_checkpoint", checkpoint, "--output_path", str(out), "--output_path_all", str(out),
        "--sd_version", "xl", "--concepts", f"{label_a}+{label_b}+{label_bg}",
        "--modifier_token", f"{token_a}+{token_b}+{token_bg}", "--resolution_h", "1024", "--resolution_w", "1024",
        "--prompt_orig", prompt, "--prompt_clean", clean, "--prompt_orig_clean", clean,
        "--seed", str(args.seed), "--t_cond", str(args.t_cond),
        "--seg_concepts", f"a {class_a}+a {class_b}", "--negative_prompt", "",
        "--seg_gpu", "-1", "--disable_xformers", "1", "--resampling_steps", str(args.resampling_steps),
        "--use_external_boxes", "1", "--external_boxes", args.external_boxes, "--boxes_only", "1",
        "--sem_binding_steps", "2", "--sem_binding_lr", "1e-4", "--sem_binding_max_steps", "1",
        "--sem_binding_use_clean", "1", "--enable_cones_mask_attention", "1", "--cones_guidance_weight", "0.08",
        "--cones_guidance_steps", "-1", "--cones_positive_value", "2.5", "--cones_negative_value", "-0.02",
        "--cones_use_sim_std", "1", "--cones_use_modifier_only", "1", "--cones_focus_fg_only", "1",
        "--cones_use_plain_prompt_before_fusion", "1", "--use_sts_kv_hook", "1", "--fusion_low_mem", "0",
    ]
    print("[infer] concepts:", args.theme1, "+", args.theme2, "+", args.background)
    print("[infer] prompt:", prompt)
    print("[infer] checkpoint:", checkpoint)
    print("[infer] command:")
    print(shlex.join(cmd))
    if not args.run:
        print("[infer] dry-run only; pass --run to execute")
        return 0
    if not args.sampler.is_file():
        raise SystemExit(f"sampler not found: {args.sampler}")
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.sampler.parent) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, cwd=args.sampler.parent, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
