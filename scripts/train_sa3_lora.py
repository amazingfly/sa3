#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def gb(value: float) -> float:
    return value / (1024**3)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def caption_metadata_fn(info: dict[str, Any], _audio: Any) -> dict[str, Any]:
    txt = Path(info["path"]).with_suffix(".txt")
    if not txt.exists():
        return {"__reject__": True}
    return {"prompt": txt.read_text(encoding="utf-8").strip()}


def import_psutil():
    try:
        import psutil  # type: ignore

        return psutil
    except ImportError as exc:
        raise RuntimeError("Missing psutil. Run ./scripts/setup_cpu.sh first.") from exc


def apply_cpu_settings(config: dict[str, Any]) -> None:
    cpu = config.get("cpu", {})
    threads = cpu.get("torch_num_threads")
    interop_threads = cpu.get("torch_num_interop_threads")

    if threads:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

    import torch  # type: ignore

    if threads:
        torch.set_num_threads(int(threads))
    if interop_threads:
        torch.set_num_interop_threads(int(interop_threads))


def patch_model_config_resolver(
    model_name: str,
    repo_id: str,
    conditioning_repo_id: str,
    work_dir: Path,
) -> None:
    from huggingface_hub import hf_hub_download  # type: ignore
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
    model_configs.models[model_name] = patched
    model_configs.base_models[model_name] = patched
    model_configs.all_models[model_name] = patched


class MemoryCallback:
    def __init__(self, config: dict[str, Any]):
        import pytorch_lightning as pl  # type: ignore

        self._base = pl.Callback
        self._callback = None
        self.config = config

    def as_callback(self):
        import pytorch_lightning as pl  # type: ignore

        outer = self

        class _MemoryCallback(pl.Callback):
            def __init__(self):
                self.psutil = import_psutil()
                self.process = self.psutil.Process(os.getpid())
                self.started_at = time.monotonic()
                self.peak_rss = 0

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                memory_cfg = outer.config.get("memory", {})
                every = int(memory_cfg.get("sample_every_steps", 1))
                step = int(trainer.global_step)
                if step % every != 0:
                    return

                rss = self.process.memory_info().rss
                self.peak_rss = max(self.peak_rss, rss)
                available = self.psutil.virtual_memory().available
                elapsed = time.monotonic() - self.started_at
                print(
                    f"memory step={step} elapsed={elapsed:.1f}s "
                    f"rss={gb(rss):.3f}GiB peak={gb(self.peak_rss):.3f}GiB "
                    f"available={gb(available):.3f}GiB",
                    flush=True,
                )

                max_rss = memory_cfg.get("max_process_rss_gb")
                min_available = memory_cfg.get("min_available_ram_gb")
                if max_rss is not None and gb(rss) > float(max_rss):
                    raise RuntimeError(f"RSS guard exceeded: {gb(rss):.3f}GiB > {max_rss}GiB")
                if min_available is not None and gb(available) < float(min_available):
                    raise RuntimeError(
                        f"available RAM guard exceeded: {gb(available):.3f}GiB < {min_available}GiB"
                    )

        return _MemoryCallback()


def dtype_from_name(name: str):
    import torch  # type: ignore

    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"fp32", "float32", "32"}:
        return torch.float32
    raise ValueError(f"Unsupported model precision: {name}")


def load_model(config: dict[str, Any], run_dir: Path):
    import torch  # type: ignore
    from safetensors.torch import load_file  # type: ignore
    from stable_audio_3.factory import create_diffusion_cond_from_config  # type: ignore
    from stable_audio_3.loading_utils import copy_state_dict  # type: ignore
    from stable_audio_3.model_configs import base_models  # type: ignore

    model_cfg = config["model"]
    model_name = model_cfg["name"]
    if model_name not in base_models:
        raise ValueError(f"LoRA training requires a base model. Got {model_name!r}; valid: {list(base_models)}")

    conditioning_repo_override = model_cfg.get("conditioning_repo_id_override")
    if conditioning_repo_override:
        patch_model_config_resolver(
            model_name=model_name,
            repo_id=model_cfg["hf_repo"],
            conditioning_repo_id=conditioning_repo_override,
            work_dir=run_dir,
        )

    local_config, local_ckpt = base_models[model_name].resolve()
    with open(local_config, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() and model_cfg.get("device") != "cpu" else "cpu")
    dtype = dtype_from_name(model_cfg.get("model_precision", "float32"))
    model = create_diffusion_cond_from_config(model_config)
    copy_state_dict(model, load_file(local_ckpt))
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    if model.pretransform is not None:
        model.pretransform.enable_grad = False
    return model, model_config, device


def merged_training_config(
    config: dict[str, Any],
    mode: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    section = dict(config["training"])
    if mode == "smoke":
        section.update(config.get("smoke", {}))
    if overrides:
        section.update({k: v for k, v in overrides.items() if v is not None})
    return section


def run_doctor(config: dict[str, Any]) -> int:
    import torch  # type: ignore

    psutil = import_psutil()
    data_dir = resolve_path(config["dataset"]["prepared_dir"])
    pairs = sorted(data_dir.glob("*.txt")) if data_dir.exists() else []
    audio = [
        p for p in data_dir.iterdir()
        if data_dir.exists() and p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus"}
    ] if data_dir.exists() else []

    print(f"Python: {sys.version.split()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Prepared dataset: {data_dir}")
    print(f"Audio files: {len(audio)}")
    print(f"Caption files: {len(pairs)}")
    vm = psutil.virtual_memory()
    print(f"RAM: total={gb(vm.total):.2f}GiB available={gb(vm.available):.2f}GiB")

    try:
        import pytorch_lightning as pl  # type: ignore
        import dill  # type: ignore

        print(f"pytorch_lightning: {pl.__version__}")
        print(f"dill: {dill.__version__}")
    except ImportError as exc:
        print(f"Missing LoRA dependency: {exc}")
        return 1

    if len(audio) != len(pairs) or not audio:
        print("Dataset is not ready. Run scripts/prepare_lora_dataset.py first.")
        return 1
    if len(audio) < 20:
        print("Warning: official SA3 LoRA docs recommend at least 20-50 clips; this dataset has fewer.")
    return 0


def train(config: dict[str, Any], mode: str, overrides: dict[str, Any] | None = None) -> Path:
    import torch  # type: ignore
    import pytorch_lightning as pl  # type: ignore
    from pytorch_lightning.loggers import CSVLogger  # type: ignore
    from stable_audio_3.data.dataset import LocalDatasetConfig, SampleDataset, collation_fn  # type: ignore
    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper  # type: ignore

    apply_cpu_settings(config)
    train_cfg = merged_training_config(config, mode, overrides)
    save_dir = resolve_path(train_cfg["save_dir"])
    run_dir = save_dir / time.strftime("%Y%m%d-%H%M%S")
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run directory: {run_dir}")
    model, model_config, device = load_model(config, run_dir)
    sample_rate = int(model.sample_rate)
    ds_ratio = int(model.pretransform.downsampling_ratio)
    sample_size = (int(float(train_cfg["duration_seconds"]) * sample_rate) // ds_ratio) * ds_ratio

    dataset = SampleDataset(
        [
            LocalDatasetConfig(
                id="train",
                path=str(resolve_path(config["dataset"]["prepared_dir"])),
                custom_metadata_fn=caption_metadata_fn,
            )
        ],
        sample_size=sample_size,
        sample_rate=sample_rate,
        force_channels="stereo",
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 1)),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        drop_last=True,
        collate_fn=collation_fn,
        worker_init_fn=lambda worker_id: torch.manual_seed(int(train_cfg.get("seed", 42)) + worker_id),
    )

    lora_config = {
        "rank": int(train_cfg.get("rank", 8)),
        "alpha": float(train_cfg.get("lora_alpha", train_cfg.get("rank", 8))),
        "adapter_type": train_cfg.get("adapter_type", "dora-rows"),
        "dropout": float(train_cfg.get("dropout", 0.0)),
        "include": train_cfg.get("include"),
        "exclude": train_cfg.get("exclude"),
    }
    optimizer_config = {
        "diffusion": {
            "optimizer": {
                "type": "AdamW",
                "config": {
                    "lr": float(train_cfg.get("lr", 1e-4)),
                    "weight_decay": 0.01,
                    "betas": [0.9, 0.95],
                },
            }
        }
    }

    pl.seed_everything(int(train_cfg.get("seed", 42)), workers=True)
    wrapper = DiffusionCondTrainingWrapper(
        model,
        mask_loss_weight=1.0,
        mask_padding_attention=True,
        silence_extension_scale_seconds=4.0,
        use_ema=False,
        log_loss_info=False,
        optimizer_configs=optimizer_config,
        pre_encoded=False,
        timestep_sampler="trunc_logit_normal",
        timestep_sampler_options={},
        inpainting_config={"mask_kwargs": {"mask_type_probabilities": [0.1, 0.8, 0.1]}},
        use_effective_length_for_schedule=True,
        sample_rate=model_config.get("sample_rate", sample_rate),
        sample_size=model_config.get("sample_size"),
        lora_config=lora_config,
        svd_bases_path=train_cfg.get("svd_bases_path"),
        log_every_n_steps=int(train_cfg.get("log_every", 1)),
        ot_coupling=False,
        base_precision=train_cfg.get("base_precision"),
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(
        every_n_train_steps=int(train_cfg.get("checkpoint_every", 50)),
        dirpath=str(checkpoint_dir),
        save_top_k=-1,
    )
    logger = CSVLogger(save_dir=str(run_dir), name="logs")
    precision = train_cfg.get("precision", "32-true" if device.type == "cpu" else "bf16-mixed")
    trainer = pl.Trainer(
        accelerator=device.type,
        devices=1,
        precision=precision,
        callbacks=[ckpt_callback, MemoryCallback(config).as_callback()],
        logger=logger,
        log_every_n_steps=1,
        max_steps=int(train_cfg["steps"]),
        default_root_dir=str(run_dir),
        gradient_clip_val=train_cfg.get("gradient_clip_val"),
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
    )

    trainer.fit(wrapper, dataloader)

    final_path = run_dir / f"cybermetal_{mode}_step{trainer.global_step}.safetensors"
    wrapper.export_lora_safetensors(final_path)
    print(f"Saved LoRA safetensors: {final_path}")
    return final_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JSON-driven Stable Audio 3 LoRA training")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "smoke", "train"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--config", default="configs/cybermetal_lora_small_cpu.json")
        if name in {"smoke", "train"}:
            sub.add_argument("--steps", type=int)
            sub.add_argument("--duration-seconds", type=float)
            sub.add_argument("--save-dir")
            sub.add_argument("--rank", type=int)
            sub.add_argument("--adapter-type")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))
    if args.command == "doctor":
        return run_doctor(config)
    overrides = {
        "steps": getattr(args, "steps", None),
        "duration_seconds": getattr(args, "duration_seconds", None),
        "save_dir": getattr(args, "save_dir", None),
        "rank": getattr(args, "rank", None),
        "adapter_type": getattr(args, "adapter_type", None),
    }
    train(config, args.command, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
