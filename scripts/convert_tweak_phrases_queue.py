#!/usr/bin/env python3
"""Convert named prompt tweaks into a native SA3 Medium Colab queue config."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_MODEL_REPO = "cocktailpeanut/stable-audio-3-medium"
DEFAULT_NEGATIVE_PROMPT = (
    "poor quality, harsh static, loud buzzing, clipping, noise, "
    "extended ambient intro, static repetitive loop, stalled arrangement, "
    "flat dynamics, silence"
)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "tweak"


def load_phrases(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = list(payload.items())
    elif isinstance(payload, list):
        items = []
        for index, item in enumerate(payload, start=1):
            if isinstance(item, str):
                items.append((f"tweak-{index:02d}", item))
            elif isinstance(item, dict):
                name = str(
                    item.get("id")
                    or item.get("name")
                    or item.get("key")
                    or f"tweak-{index:02d}"
                )
                phrase = item.get("prompt") or item.get("phrase") or item.get("text")
                items.append((name, phrase))
            else:
                raise ValueError(f"Unsupported list item at index {index - 1}")
    else:
        raise ValueError("Input must be a JSON object or list")

    phrases = []
    for name, phrase in items:
        phrase_text = str(phrase or "").strip()
        if not phrase_text:
            raise ValueError(f"Phrase for {name!r} is empty")
        phrases.append((slugify(str(name)), phrase_text))
    if not phrases:
        raise ValueError("No phrases found")
    return phrases


def load_base_prompts(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = payload.get("prompts") or payload.get("items") or payload.get("tracks")
        if items is None:
            items = [
                {"title": key, "prompt": value}
                for key, value in payload.items()
            ]
    else:
        raise ValueError("Base prompts input must be a JSON object or list")

    if not isinstance(items, list):
        raise ValueError("Base prompts must be a list")

    prompts = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, str):
            title = f"prompt-{index:02d}"
            prompt = item
        elif isinstance(item, dict):
            title = str(
                item.get("title")
                or item.get("id")
                or item.get("name")
                or f"prompt-{index:02d}"
            )
            prompt = item.get("prompt") or item.get("text")
        else:
            raise ValueError(f"Unsupported base prompt item at index {index - 1}")

        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            raise ValueError(f"Base prompt for {title!r} is empty")
        prompts.append((slugify(title), prompt_text))

    if not prompts:
        raise ValueError("No base prompts found")
    return prompts


def combined_prompt(base_prompt: str, phrase: str, append_template: str) -> str:
    if "{phrase}" in append_template:
        appended = append_template.format(phrase=phrase)
    else:
        appended = f"{append_template}{phrase}"
    return f"{base_prompt.rstrip()}{appended}"


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    phrases = load_phrases(Path(args.input))
    base_prompts = (
        load_base_prompts(Path(args.base_prompts))
        if args.base_prompts
        else [(None, "")]
    )
    generations = []
    for base_index, (base_name, base_prompt) in enumerate(base_prompts, start=1):
        for phrase_index, (phrase_name, phrase) in enumerate(phrases, start=1):
            index = len(generations)
            if base_name is None:
                generation_id = phrase_name
                output_stem = f"{args.output_prefix}-{phrase_name}"
                prompt = " ".join(
                    part
                    for part in (args.prompt_prefix, phrase, args.prompt_suffix)
                    if part
                )
            else:
                generation_id = (
                    f"{base_index:02d}-{base_name}-"
                    f"{phrase_index:02d}-{phrase_name}"
                )
                output_stem = f"{args.output_prefix}-{generation_id}"
                prompt = combined_prompt(base_prompt, phrase, args.append_template)

            generations.append(
                {
                    "id": generation_id,
                    "duration_seconds": args.duration_seconds,
                    "steps": args.steps,
                    "cfg_scale": args.cfg_scale,
                    "apg_scale": args.apg_scale,
                    "duration_padding_seconds": args.duration_padding_seconds,
                    "sampler_type": args.sampler_type,
                    "seed": args.seed_start + index,
                    "prompt": prompt,
                    "negative_prompt": args.negative_prompt,
                    "output_name": f"{output_stem}.flac",
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
        "generations": generations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert tweak phrases into a queued SA3 Medium Colab config."
    )
    parser.add_argument("input", help="Input JSON, for example ~/Downloads/tweak_phrases.json")
    parser.add_argument(
        "--base-prompts",
        default="",
        help="Optional JSON containing base title/prompt objects to combine with each tweak",
    )
    parser.add_argument(
        "--output",
        default="configs/tweak_phrases_medium_queue.json",
        help="Output queue config path",
    )
    parser.add_argument("--duration-seconds", type=float, default=380.0)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--apg-scale", type=float, default=1.0)
    parser.add_argument("--duration-padding-seconds", type=float, default=6.0)
    parser.add_argument("--sampler-type", default="pingpong")
    parser.add_argument("--seed-start", type=int, default=380140)
    parser.add_argument("--output-prefix", default="tweak-medium-380s")
    parser.add_argument("--flac-subtype", default="PCM_24")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO)
    parser.add_argument("--conditioning-repo", default="")
    parser.add_argument("--prompt-prefix", default="")
    parser.add_argument("--prompt-suffix", default="")
    parser.add_argument(
        "--append-template",
        default="\n\nArrangement tweak: {phrase}",
        help="Template appended to each base prompt; use {phrase} for the tweak text",
    )
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = build_config(args)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(config['generations'])} queued generations to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
