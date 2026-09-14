#!/usr/bin/env python3
"""Build a native SA3 Medium queue from named prompts and a short CFG sweep."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL_REPO = "cocktailpeanut/stable-audio-3-medium"
DEFAULT_NEGATIVE_PROMPT = (
    "poor quality, harsh static, loud buzzing, clipping, noise, "
    "extended ambient intro, long intro, stalled arrangement, "
    "static repetitive loop, flat dynamics, silence"
)


def slugify(value: str, *, max_len: int = 80) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-") or "prompt"
    return value[:max_len].strip("-") or "prompt"


def parse_cfg_values(value: str) -> list[float]:
    cfgs = [round(float(part.strip()), 3) for part in value.split(",") if part.strip()]
    if not cfgs:
        raise ValueError("--cfg-values must contain at least one number")
    return cfgs


def load_prompts(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Prompt list item {index} must be an object")
            name = str(item.get("title") or item.get("name") or item.get("id") or f"prompt-{index:02d}")
            prompt = item.get("prompt")
            items.append((name, prompt))
    else:
        raise ValueError("Prompt input must be a JSON object or list")

    prompts: list[tuple[str, str]] = []
    for name, prompt in items:
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            raise ValueError(f"Prompt for {name!r} is empty")
        prompts.append((str(name), prompt_text))
    return prompts


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(Path(args.input))
    cfgs = parse_cfg_values(args.cfg_values)
    generations: list[dict[str, Any]] = []

    for cfg_index, cfg in enumerate(cfgs, start=1):
        for prompt_index, (name, prompt) in enumerate(prompts, start=1):
            queue_index = len(generations)
            name_slug = slugify(name)
            generation_id = f"g{cfg_index:02d}-p{prompt_index:02d}-{name_slug}"
            generations.append(
                {
                    "id": generation_id,
                    "duration_seconds": args.duration_seconds,
                    "steps": args.steps,
                    "cfg_scale": cfg,
                    "apg_scale": args.apg_scale,
                    "duration_padding_seconds": args.duration_padding_seconds,
                    "sampler_type": args.sampler_type,
                    "seed": args.seed_start + queue_index,
                    "prompt": prompt,
                    "negative_prompt": args.negative_prompt,
                    "output_name": f"{args.output_prefix}-{generation_id}.flac",
                    "flac_subtype": args.flac_subtype,
                }
            )

    return {
        "queue_version": 1,
        "model": {
            "name": "medium",
            "repo": args.model_repo,
            "conditioning_repo": args.conditioning_repo or args.model_repo,
            "model_half": True,
            "low_memory_load": True,
            "chunked_decode": True,
            "sequential_chunked_decode": False,
            "allow_turing_sdpa_fallback": True,
            "max_duration_seconds": 380,
        },
        "metadata": {
            "source_prompt_file": str(Path(args.input).expanduser()),
            "source_prompt_count": len(prompts),
            "cfg_values": cfgs,
            "queue_item_count": len(generations),
            "prompt_order": [name for name, _ in prompts],
        },
        "generations": generations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Medium Colab guidance-sweep queue from named prompts."
    )
    parser.add_argument("input", help="JSON object or list containing named prompts")
    parser.add_argument("--output", default="configs/prompt_guidance_sweep_medium_queue.json")
    parser.add_argument("--output-prefix", default="promptguidesweep-medium-380s")
    parser.add_argument("--cfg-values", default="1.0,0.95,0.9")
    parser.add_argument("--duration-seconds", type=float, default=380.0)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--apg-scale", type=float, default=1.0)
    parser.add_argument("--duration-padding-seconds", type=float, default=6.0)
    parser.add_argument("--sampler-type", default="pingpong")
    parser.add_argument("--seed-start", type=int, default=706000)
    parser.add_argument("--flac-subtype", default="PCM_24")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--conditioning-repo", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    config = build_config(args)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(config['generations'])} queued generations from "
        f"{config['metadata']['source_prompt_count']} prompts to {output}"
    )
    print(f"CFG values: {config['metadata']['cfg_values']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
