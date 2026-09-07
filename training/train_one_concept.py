#!/usr/bin/env python3
"""Build or run one concept-personalization training job.

The entry point exposes the paper training implementation with explicit,
auditable arguments. It prints the command by default and executes with --run.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DEFAULT_SCRIPT = PROJECT / "source" / "training" / "train_personalized_concept.py"


def build_command(args: argparse.Namespace) -> list[str]:
    cmd = [
        args.launcher,
        "launch",
        "--num_processes",
        "1",
        str(args.training_script),
        f"--pretrained_model_name_or_path={args.model}",
        f"--instance_data_dir={args.dataset_dir}",
        f"--output_dir={args.output_dir}",
        f"--instance_prompt={args.instance_prompt}",
        f"--resolution={args.resolution}",
        f"--train_batch_size={args.batch_size}",
        f"--learning_rate={args.learning_rate}",
        f"--lr_warmup_steps={args.lr_warmup_steps}",
        f"--max_train_steps={args.max_train_steps}",
        f"--num_class_images={args.num_class_images}",
        f"--save_steps={args.save_steps}",
        f"--modifier_token={args.modifier_token}",
        f"--gradient_accumulation_steps={args.gradient_accumulation_steps}",
    ]
    for flag in ("--scale_lr", "--hflip", "--use_8bit_adam", "--gradient_checkpointing"):
        if getattr(args, flag[2:].replace("-", "_")):
            cmd.append(flag)
    return cmd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--concept-id", default="subject_a")
    p.add_argument("--dataset-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--instance-prompt", default="Photo of a <subject_a> object.")
    p.add_argument("--modifier-token", default="<subject_a>")
    p.add_argument("--model", default=os.environ.get("MODEL_NAME", "/path/to/models/stable-diffusion-xl-base-1.0"))
    p.add_argument("--training-script", type=Path, default=DEFAULT_SCRIPT)
    p.add_argument("--launcher", default="accelerate")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    p.add_argument("--max-train-steps", type=int, default=1500)
    p.add_argument("--num-class-images", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=200)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--run", action="store_true", help="actually launch accelerate; default is dry-run")
    p.add_argument("--no-scale-lr", dest="scale_lr", action="store_false")
    p.add_argument("--no-hflip", dest="hflip", action="store_false")
    p.add_argument("--no-8bit-adam", dest="use_8bit_adam", action="store_false")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.set_defaults(scale_lr=True, hflip=True, use_8bit_adam=True, gradient_checkpointing=True)
    args = p.parse_args()
    if args.dataset_dir is None:
        args.dataset_dir = Path(os.environ.get("DATASET_BASE", "/path/to/datasets")) / args.concept_id
    if args.output_dir is None:
        args.output_dir = PROJECT / "runs" / "train" / args.concept_id
    args.dataset_dir = args.dataset_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.training_script = args.training_script.resolve()
    return args


def main() -> int:
    args = parse_args()
    cmd = build_command(args)
    print("[train] command:")
    print(shlex.join(cmd))
    print(f"[train] dataset exists: {args.dataset_dir.exists()} ({args.dataset_dir})")
    print(f"[train] training script exists: {args.training_script.exists()} ({args.training_script})")
    if not args.run:
        print("[train] dry-run only; pass --run to execute")
        return 0
    if not args.dataset_dir.is_dir():
        raise SystemExit(f"dataset directory not found: {args.dataset_dir}")
    if not args.training_script.is_file():
        raise SystemExit(f"training script not found: {args.training_script}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["PYTHONPATH"] = str(args.training_script.parent) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.call(cmd, cwd=args.training_script.parent, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
