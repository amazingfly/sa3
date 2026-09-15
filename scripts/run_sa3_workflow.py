#!/usr/bin/env python3
from __future__ import annotations

# Load centralized workstation defaults; explicit environment/CLI values win.
import sys as _workspace_sys
from pathlib import Path as _WorkspacePath
for _workspace_root in _WorkspacePath(__file__).resolve().parents:
    if (_workspace_root / "media_workspace").is_dir():
        _workspace_sys.path.insert(0, str(_workspace_root))
        break
from media_workspace.config import apply_environment as _apply_workspace
_apply_workspace()


import argparse
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def gb(value: float) -> float:
    return value / (1024**3)


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "track"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = ["model", "defaults", "prompts"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    model = config["model"]
    if model.get("device", "cpu") != "cpu":
        raise ValueError("This workflow is intended for CPU-only generation; set model.device to cpu.")

    max_duration = float(model.get("max_duration_seconds", 120))
    if model.get("name") == "small-music" and max_duration > 120:
        raise ValueError("small-music supports up to 120 seconds; reduce model.max_duration_seconds.")

    prompt_ids = set()
    for prompt in config["prompts"]:
        if "id" not in prompt or "prompt" not in prompt:
            raise ValueError("Each prompt must have id and prompt.")
        prompt_ids.add(prompt["id"])

    probe = config.get("probe", {})
    if probe.get("prompt_id") and probe["prompt_id"] not in prompt_ids:
        raise ValueError(f"probe.prompt_id does not match any prompt: {probe['prompt_id']}")

    for section in ("smoke", "probe", "full_generation"):
        if section not in config:
            continue
        durations = config[section].get("durations_seconds")
        if durations is None and "duration_seconds" in config[section]:
            durations = [config[section]["duration_seconds"]]
        if durations:
            too_long = [d for d in durations if float(d) > max_duration]
            if too_long:
                raise ValueError(
                    f"{section} contains durations above max_duration_seconds={max_duration}: {too_long}"
                )


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(config: dict[str, Any], label: str) -> Path:
    base = ROOT / config.get("output_dir", "outputs")
    run_dir = base / f"{timestamp()}-{label}"
    (run_dir / "jobs").mkdir(parents=True, exist_ok=False)
    (run_dir / "audio").mkdir(parents=True, exist_ok=True)
    return run_dir


def import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except ImportError as exc:
        raise RuntimeError("Missing psutil. Run ./scripts/setup_cpu.sh first.") from exc


def run_doctor(config: dict[str, Any]) -> int:
    psutil = import_psutil()
    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU count: {os.cpu_count()}")
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    print(f"RAM: total={gb(vm.total):.2f} GiB available={gb(vm.available):.2f} GiB")
    print(f"Swap: total={gb(swap.total):.2f} GiB used={gb(swap.used):.2f} GiB")

    model = config["model"]
    print(f"Model: {model['name']} ({model.get('hf_repo', 'unknown repo')}) on {model.get('device', 'cpu')}")
    print(f"CPU torch threads: {config.get('cpu', {}).get('torch_num_threads', 'default')}")
    print(f"HF_TOKEN: {'set' if os.environ.get('HF_TOKEN') else 'unset'}")
    print(f"HUGGINGFACE_HUB_TOKEN: {'set' if os.environ.get('HUGGINGFACE_HUB_TOKEN') else 'unset'}")
    saved_token_present = False
    hf_access_ok = False
    try:
        from huggingface_hub import get_token, hf_hub_download  # type: ignore

        saved_token_present = bool(get_token())
        print(f"Saved Hugging Face token: {'set' if saved_token_present else 'unset'}")
        try:
            hf_hub_download(
                repo_id=model.get("hf_repo", "stabilityai/stable-audio-3-small-music"),
                filename="model_config.json",
                dry_run=True,
                token=True,
            )
            hf_access_ok = True
            print("Model file access: ok")
        except Exception as access_exc:
            print(f"Model file access: failed ({access_exc.__class__.__name__}: {access_exc})")
    except Exception as exc:
        print(f"Saved Hugging Face token: unknown ({exc})")

    try:
        import torch  # type: ignore
        import torchaudio  # type: ignore

        print(f"torch: {torch.__version__}")
        print(f"torchaudio: {torchaudio.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError as exc:
        print(f"PyTorch import failed: {exc}")
        return 1

    try:
        import stable_audio_3  # type: ignore  # noqa: F401

        print("stable_audio_3: import ok")
    except ImportError as exc:
        print(f"stable_audio_3 import failed: {exc}")
        return 1

    token_present = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or saved_token_present
    )
    if not token_present:
        print("Warning: Hugging Face token is not set. The model is gated and generation will fail until access is configured.")
    elif not hf_access_ok:
        print("Warning: Hugging Face token is present but is not authorized for the configured gated model.")

    return 0 if hf_access_ok else 1


@dataclass
class Job:
    kind: str
    prompt_id: str
    prompt: str
    output_path: Path
    duration_seconds: float
    steps: int
    cfg_scale: float
    seed: int
    negative_prompt: str | None
    chunked_decode: bool
    apg_scale: float
    duration_padding_sec: float
    sampler_type: str | None

    def to_payload(self, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "model": config["model"],
            "cpu": config.get("cpu", {}),
            "job": {
                "kind": self.kind,
                "prompt_id": self.prompt_id,
                "prompt": self.prompt,
                "output_path": str(self.output_path),
                "duration_seconds": self.duration_seconds,
                "steps": self.steps,
                "cfg_scale": self.cfg_scale,
                "seed": self.seed,
                "negative_prompt": self.negative_prompt,
                "chunked_decode": self.chunked_decode,
                "apg_scale": self.apg_scale,
                "duration_padding_sec": self.duration_padding_sec,
                "sampler_type": self.sampler_type,
            },
        }


def prompt_by_id(config: dict[str, Any], prompt_id: str) -> dict[str, str]:
    for prompt in config["prompts"]:
        if prompt["id"] == prompt_id:
            return prompt
    raise KeyError(prompt_id)


def defaults_for(config: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    defaults = dict(config.get("defaults", {}))
    defaults.update({k: v for k, v in section.items() if v is not None})
    return defaults


def build_job(
    config: dict[str, Any],
    run_dir: Path,
    kind: str,
    prompt: dict[str, str],
    duration: float,
    settings: dict[str, Any],
    seed: int,
    suffix: str,
) -> Job:
    audio_format = str(config.get("audio_format", "wav")).lower().lstrip(".")
    if audio_format not in {"wav", "flac"}:
        raise ValueError(f"Unsupported audio_format: {audio_format}. Use wav or flac.")
    audio_name = f"{kind}-{slugify(prompt['id'])}-{int(duration)}s-{suffix}.{audio_format}"
    model = config["model"]
    return Job(
        kind=kind,
        prompt_id=prompt["id"],
        prompt=prompt["prompt"],
        output_path=run_dir / "audio" / audio_name,
        duration_seconds=float(duration),
        steps=int(settings.get("steps", config["defaults"].get("steps", 8))),
        cfg_scale=float(settings.get("cfg_scale", config["defaults"].get("cfg_scale", 1.0))),
        seed=int(seed),
        negative_prompt=settings.get("negative_prompt"),
        chunked_decode=bool(model.get("chunked_decode", True)),
        apg_scale=float(settings.get("apg_scale", config["defaults"].get("apg_scale", 1.0))),
        duration_padding_sec=float(
            settings.get("duration_padding_sec", config["defaults"].get("duration_padding_sec", 6.0))
        ),
        sampler_type=settings.get("sampler_type"),
    )


def settings_with_prompt_overrides(settings: dict[str, Any], prompt: dict[str, Any]) -> dict[str, Any]:
    merged = dict(settings)
    for key in (
        "steps",
        "cfg_scale",
        "apg_scale",
        "duration_padding_sec",
        "sampler_type",
        "negative_prompt",
    ):
        if key in prompt and prompt[key] is not None:
            merged[key] = prompt[key]
    return merged


def monitor_process(proc: subprocess.Popen[Any], interval_seconds: float) -> dict[str, Any]:
    psutil = import_psutil()
    process = psutil.Process(proc.pid)
    peak_rss = 0
    min_available = math.inf
    samples = 0
    start = time.monotonic()

    def collect() -> None:
        nonlocal peak_rss, min_available, samples
        rss = 0
        try:
            procs = [process] + process.children(recursive=True)
            for child in procs:
                try:
                    rss += child.memory_info().rss
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
        peak_rss = max(peak_rss, rss)
        min_available = min(min_available, psutil.virtual_memory().available)
        samples += 1

    while proc.poll() is None:
        collect()
        time.sleep(interval_seconds)
    collect()

    return {
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "peak_process_rss_gb": round(gb(float(peak_rss)), 3),
        "min_system_available_gb": round(gb(float(min_available)), 3),
        "samples": samples,
    }


def run_job(config: dict[str, Any], run_dir: Path, job: Job) -> dict[str, Any]:
    job_dir = run_dir / "jobs"
    suffix = f"{job.kind}-{slugify(job.prompt_id)}-{int(job.duration_seconds)}s-seed{job.seed}"
    payload_path = job_dir / f"{suffix}.json"
    log_path = job_dir / f"{suffix}.log"
    payload = job.to_payload(config)
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    env = os.environ.copy()
    cpu = config.get("cpu", {})
    if cpu.get("torch_num_threads"):
        env["OMP_NUM_THREADS"] = str(cpu["torch_num_threads"])
        env["MKL_NUM_THREADS"] = str(cpu["torch_num_threads"])
    env["TOKENIZERS_PARALLELISM"] = "false"

    print(
        f"Starting {job.kind}: prompt={job.prompt_id} duration={job.duration_seconds:g}s "
        f"steps={job.steps} cfg={job.cfg_scale} seed={job.seed}"
        ,
        flush=True,
    )

    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "_worker", "--payload", str(payload_path)],
            cwd=str(ROOT),
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        memory = monitor_process(
            proc, float(config.get("memory", {}).get("sample_interval_seconds", 0.5))
        )
        return_code = proc.wait()

    result = {
        "job": payload["job"],
        "return_code": return_code,
        "ok": return_code == 0 and job.output_path.exists() and job.output_path.stat().st_size > 0,
        "output_path": str(job.output_path),
        "payload_path": str(payload_path),
        "log_path": str(log_path),
        "memory": memory,
    }

    guard = config.get("memory", {})
    max_rss = guard.get("max_process_rss_gb")
    min_available = guard.get("min_available_ram_gb")
    if max_rss is not None and memory["peak_process_rss_gb"] > float(max_rss):
        result["ok"] = False
        result["guard_failure"] = f"peak RSS {memory['peak_process_rss_gb']} GiB exceeded {max_rss} GiB"
    if min_available is not None and memory["min_system_available_gb"] < float(min_available):
        result["ok"] = False
        result["guard_failure"] = (
            f"available RAM {memory['min_system_available_gb']} GiB fell below {min_available} GiB"
        )

    status = "ok" if result["ok"] else "failed"
    print(
        f"Finished {job.kind}: {status}, elapsed={memory['elapsed_seconds']}s, "
        f"peak_rss={memory['peak_process_rss_gb']} GiB, "
        f"min_available={memory['min_system_available_gb']} GiB"
        ,
        flush=True,
    )
    if not result["ok"]:
        print(f"  log: {log_path}", flush=True)
        if result.get("guard_failure"):
            print(f"  guard: {result['guard_failure']}", flush=True)
    return result


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def run_smoke(config: dict[str, Any], run_dir: Path, manifest: dict[str, Any]) -> bool:
    smoke_cfg = config.get("smoke", {})
    settings = defaults_for(config, smoke_cfg)
    prompt = config["prompts"][0]
    job = build_job(
        config,
        run_dir,
        "smoke",
        prompt,
        float(smoke_cfg.get("duration_seconds", 5)),
        settings,
        int(smoke_cfg.get("seed", 260527)),
        "smoke",
    )
    result = run_job(config, run_dir, job)
    manifest.setdefault("smoke", []).append(result)
    write_manifest(run_dir, manifest)
    return bool(result["ok"])


def run_probe(config: dict[str, Any], run_dir: Path, manifest: dict[str, Any]) -> float | None:
    probe_cfg = config.get("probe", {})
    settings = defaults_for(config, probe_cfg)
    prompt = prompt_by_id(config, probe_cfg.get("prompt_id", config["prompts"][0]["id"]))
    repeats = int(config.get("memory", {}).get("probe_repeats", 2))
    durations = [float(d) for d in probe_cfg.get("durations_seconds", [5, 15, 30, 60, 90, 120])]
    seed_start = int(probe_cfg.get("seed_start", 261000))

    largest_stable: float | None = None
    probe_results = manifest.setdefault("probe", [])

    for duration in durations:
        duration_ok = True
        for repeat in range(repeats):
            seed = seed_start + int(duration * 10) + repeat
            job = build_job(
                config,
                run_dir,
                "probe",
                prompt,
                duration,
                settings,
                seed,
                f"r{repeat + 1}",
            )
            result = run_job(config, run_dir, job)
            probe_results.append(result)
            write_manifest(run_dir, manifest)
            if not result["ok"]:
                duration_ok = False
                break
        if duration_ok:
            largest_stable = duration
            manifest["largest_stable_duration_seconds"] = largest_stable
            write_manifest(run_dir, manifest)
        else:
            print(f"Stopping probe at {duration:g}s; largest stable duration is {largest_stable}.")
            break

    return largest_stable


def run_full(
    config: dict[str, Any],
    run_dir: Path,
    manifest: dict[str, Any],
    largest_stable_duration: float | None,
) -> bool:
    full_cfg = config.get("full_generation", {})
    if not full_cfg.get("enabled", True):
        print("Full generation disabled in config.", flush=True)
        return True

    settings = defaults_for(config, full_cfg)
    configured_durations = [float(d) for d in full_cfg.get("durations_seconds", [30, 60, 90, 120])]
    if full_cfg.get("use_largest_stable_duration", True) and largest_stable_duration:
        durations = [d for d in configured_durations if d <= largest_stable_duration]
        if largest_stable_duration not in durations:
            durations.append(largest_stable_duration)
        durations = sorted(set(durations))
    else:
        durations = configured_durations

    if not durations:
        print("No full-generation durations remain after applying stability probe limit.", flush=True)
        return False

    seed = int(full_cfg.get("seed_start", 270000))
    variations = int(full_cfg.get("variations_per_prompt", 2))
    stop_on_first_failure = bool(full_cfg.get("stop_on_first_failure", False))
    all_ok = True
    full_results = manifest.setdefault("full_generation", [])

    for prompt in config["prompts"]:
        prompt_settings = settings_with_prompt_overrides(settings, prompt)
        for duration in durations:
            for variation in range(variations):
                job = build_job(
                    config,
                    run_dir,
                    "full",
                    prompt,
                    duration,
                    prompt_settings,
                    seed,
                    f"v{variation + 1}",
                )
                seed += 1
                result = run_job(config, run_dir, job)
                full_results.append(result)
                write_manifest(run_dir, manifest)
                if not result["ok"]:
                    all_ok = False
                    if stop_on_first_failure:
                        return False
    return all_ok


def run_command(command: str, config: dict[str, Any]) -> int:
    run_dir = make_run_dir(config, command)
    manifest: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": command,
        "config": config,
        "run_dir": str(run_dir),
        "results": {},
    }
    write_manifest(run_dir, manifest)
    print(f"Run directory: {run_dir}")

    if command == "smoke":
        ok = run_smoke(config, run_dir, manifest)
        return 0 if ok else 1

    if command == "probe":
        largest = run_probe(config, run_dir, manifest)
        return 0 if largest else 1

    if command == "full":
        ok = run_full(config, run_dir, manifest, None)
        return 0 if ok else 1

    if command == "all":
        if not run_smoke(config, run_dir, manifest):
            print("Smoke generation failed; skipping probe and full generation.")
            return 1
        largest = run_probe(config, run_dir, manifest)
        if not largest:
            print("No stable probe duration found; skipping full generation.")
            return 1
        ok = run_full(config, run_dir, manifest, largest)
        return 0 if ok else 1

    raise ValueError(f"Unhandled command: {command}")


def worker(payload_path: Path) -> int:
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    model_cfg = payload["model"]
    cpu_cfg = payload.get("cpu", {})
    job = payload["job"]

    threads = cpu_cfg.get("torch_num_threads")
    interop_threads = cpu_cfg.get("torch_num_interop_threads")

    import torch  # type: ignore
    import torchaudio  # type: ignore
    from stable_audio_3 import StableAudioModel  # type: ignore

    if threads:
        torch.set_num_threads(int(threads))
    if interop_threads:
        torch.set_num_interop_threads(int(interop_threads))

    output_path = Path(job["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conditioning_repo_override = model_cfg.get("conditioning_repo_id_override")
    if conditioning_repo_override:
        patch_model_config_resolver(
            model_name=model_cfg["name"],
            repo_id=model_cfg["hf_repo"],
            conditioning_repo_id=conditioning_repo_override,
            work_dir=payload_path.parent,
        )

    print(f"Loading model {model_cfg['name']} on {model_cfg.get('device', 'cpu')}")
    model = StableAudioModel.from_pretrained(
        model_cfg["name"],
        device=model_cfg.get("device", "cpu"),
        model_half=bool(model_cfg.get("model_half", False)),
    )
    lora_paths = model_cfg.get("lora_ckpt_paths", [])
    if lora_paths:
        resolved_lora_paths = [
            str((ROOT / path).resolve() if not Path(path).is_absolute() else Path(path))
            for path in lora_paths
        ]
        print(f"Loading LoRA checkpoints: {resolved_lora_paths}")
        model.load_lora(resolved_lora_paths)
        if model_cfg.get("lora_strength") is not None:
            model.set_lora_strength(float(model_cfg["lora_strength"]))

    kwargs: dict[str, Any] = {
        "prompt": job["prompt"],
        "negative_prompt": job.get("negative_prompt"),
        "duration": float(job["duration_seconds"]),
        "steps": int(job["steps"]),
        "cfg_scale": float(job["cfg_scale"]),
        "seed": int(job["seed"]),
        "batch_size": 1,
        "chunked_decode": bool(job.get("chunked_decode", True)),
        "apg_scale": float(job.get("apg_scale", 1.0)),
        "duration_padding_sec": float(job.get("duration_padding_sec", 6.0)),
    }
    if job.get("sampler_type"):
        kwargs["sampler_type"] = job["sampler_type"]

    print(
        f"Generating {job['duration_seconds']}s, steps={job['steps']}, "
        f"cfg={job['cfg_scale']}, seed={job['seed']}"
    )
    audio = model.generate(**kwargs)
    audio = audio.detach().to(torch.float32).cpu()
    if audio.dim() == 3:
        audio_to_save = audio[0]
    elif audio.dim() == 2:
        audio_to_save = audio
    else:
        raise ValueError(f"Unexpected audio tensor shape: {tuple(audio.shape)}")

    torchaudio.save(str(output_path), audio_to_save, model.model.sample_rate)
    print(f"Saved {output_path}")
    return 0


def patch_model_config_resolver(
    model_name: str,
    repo_id: str,
    conditioning_repo_id: str,
    work_dir: Path,
) -> None:
    from huggingface_hub import hf_hub_download  # type: ignore
    import stable_audio_3.model as model_module  # type: ignore
    import stable_audio_3.model_configs as model_configs  # type: ignore

    class PatchedModelConfig:
        def resolve(self):
            config_path = hf_hub_download(repo_id=repo_id, filename="model_config.json", token=True)
            ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors", token=True)
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            for cond in config.get("model", {}).get("conditioning", {}).get("configs", []):
                cond_config = cond.get("config", {})
                if cond_config.get("repo_id"):
                    cond_config["repo_id"] = conditioning_repo_id

            patched_path = work_dir / f"{model_name}-patched-model_config.json"
            patched_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
            return str(patched_path), ckpt_path

    patched = PatchedModelConfig()
    model_configs.all_models[model_name] = patched
    model_module.all_models[model_name] = patched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stable Audio 3 Small CPU JSON workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("doctor", "smoke", "probe", "full", "all"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="configs/cyberpunk_industrial.json")

    worker_parser = subparsers.add_parser("_worker")
    worker_parser.add_argument("--payload", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "_worker":
        return worker(Path(args.payload))

    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_config(config_path)
    if args.command == "doctor":
        return run_doctor(config)
    return run_command(args.command, config)


if __name__ == "__main__":
    raise SystemExit(main())
