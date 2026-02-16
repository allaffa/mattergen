from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel
import yaml

from mattergen.common.data.dataloader import build_split_dataloader
from mattergen.common.data.property_scalers import compute_property_scalers
from mattergen.diffusion.data.batched_data import BatchedData
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.training_components import (
    OptimizerPartial,
    SchedulerPartial,
    build_optimizers_and_schedulers,
    calc_loss,
)

logger = logging.getLogger(__name__)


def _is_distributed_env() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def _is_rank_zero(rank: int) -> bool:
    return rank == 0


def _resolve_device(local_rank: int | None) -> torch.device:
    if torch.cuda.is_available():
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
            return torch.device("cuda", local_rank)
        return torch.device("cuda")
    return torch.device("cpu")


def _to_device(batch: Any, device: torch.device):
    if hasattr(batch, "to"):
        return batch.to(device)
    return batch


def _mean_reduce(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if distributed:
        value = value.detach().clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


def _parse_optimizers(configured: Any) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    if isinstance(configured, torch.optim.Optimizer):
        return configured, []

    if isinstance(configured, (list, tuple)):
        if len(configured) == 2 and isinstance(configured[0], list):
            optimizers = configured[0]
            scheduler_cfgs = configured[1]
            if len(optimizers) != 1:
                raise ValueError("Native DDP scaffold supports exactly one optimizer.")
            return optimizers[0], scheduler_cfgs

    raise ValueError(f"Unsupported optimizer/scheduler configuration type: {type(configured)}")


def _step_schedulers(
    scheduler_cfgs: list[dict[str, Any]],
    when: str,
    val_loss: float | None = None,
):
    for scheduler_cfg in scheduler_cfgs:
        scheduler = scheduler_cfg["scheduler"]
        interval = scheduler_cfg.get("interval", "epoch")
        if interval != when:
            continue

        monitor_key = scheduler_cfg.get("monitor")
        if monitor_key is not None and val_loss is not None:
            scheduler.step(val_loss)
        else:
            scheduler.step()


def _extract_checkpoint_cfg(trainer_cfg: DictConfig) -> dict[str, Any]:
    checkpoint_cfg = trainer_cfg.get("checkpoint")
    if checkpoint_cfg is not None:
        return {
            "monitor": checkpoint_cfg.get("monitor", "loss_val"),
            "mode": checkpoint_cfg.get("mode", "min"),
            "save_top_k": int(checkpoint_cfg.get("save_top_k", 1)),
            "save_last": bool(checkpoint_cfg.get("save_last", True)),
            "every_n_epochs": int(checkpoint_cfg.get("every_n_epochs", 1)),
            "filename": checkpoint_cfg.get("filename", "{epoch}-{loss_val:.2f}"),
        }

    # Legacy Lightning-style callback config fallback.
    callbacks = trainer_cfg.get("callbacks", [])
    for callback_cfg in callbacks:
        if "ModelCheckpoint" in str(callback_cfg.get("_target_", "")):
            return {
                "monitor": callback_cfg.get("monitor", "loss_val"),
                "mode": callback_cfg.get("mode", "min"),
                "save_top_k": int(callback_cfg.get("save_top_k", 1)),
                "save_last": bool(callback_cfg.get("save_last", True)),
                "every_n_epochs": int(callback_cfg.get("every_n_epochs", 1)),
                "filename": callback_cfg.get("filename", "{epoch}-{loss_val:.2f}"),
            }
    return {
        "monitor": "loss_val",
        "mode": "min",
        "save_top_k": 1,
        "save_last": True,
        "every_n_epochs": 1,
        "filename": "{epoch}-{loss_val:.2f}",
    }


def _extract_logger_cfg(trainer_cfg: DictConfig) -> dict[str, Any] | None:
    logger_cfg = trainer_cfg.get("logger")
    if logger_cfg is None:
        return None

    logger_type = str(logger_cfg.get("type", "")).lower()
    if logger_type == "wandb":
        return {
            "project": logger_cfg.get("project"),
            "job_type": logger_cfg.get("job_type"),
        }

    # Legacy Lightning-style logger config fallback.
    target = str(logger_cfg.get("_target_", ""))
    if "WandbLogger" not in target:
        return None
    return {
        "project": logger_cfg.get("project"),
        "job_type": logger_cfg.get("job_type"),
    }


def _format_checkpoint_name(pattern: str, epoch: int, metric_name: str, metric_value: float | None) -> str:
    metric_value = float("nan") if metric_value is None else metric_value
    out = pattern.replace("{epoch}", str(epoch))
    regex = re.compile(r"\{(" + re.escape(metric_name) + r")(:[^}]*)?\}")

    def repl(match: re.Match) -> str:
        fmt = match.group(2)
        if fmt:
            return format(metric_value, fmt[1:])
        return f"{metric_value}"

    return regex.sub(repl, out)


def _load_checkpoint(
    ckpt_path: str,
    diffusion_module: DiffusionModule[BatchedData],
    optimizer: torch.optim.Optimizer,
    scheduler_cfgs: list[dict[str, Any]],
) -> tuple[int, float | None]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    diffusion_module.load_state_dict(checkpoint["state_dict"], strict=True)

    optimizer_states = checkpoint.get("optimizer_states", [])
    if optimizer_states:
        optimizer.load_state_dict(optimizer_states[0])

    scheduler_states = checkpoint.get("scheduler_states", [])
    for scheduler_cfg, scheduler_state in zip(scheduler_cfgs, scheduler_states):
        scheduler_cfg["scheduler"].load_state_dict(scheduler_state)

    next_epoch = int(checkpoint.get("epoch", -1)) + 1
    best_metric = checkpoint.get("best_metric")
    return next_epoch, best_metric


def _save_config_yaml(config_dict: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config_dict, f)


def _save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float | None,
    config_dict: dict[str, Any],
    scheduler_cfgs: list[dict[str, Any]],
    ckpt_cfg: dict[str, Any],
    best_k: list[tuple[float, Path]],
    best_metric: float,
):
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metric_name = str(ckpt_cfg["monitor"])
    metric_value = val_loss
    ckpt_payload = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer_states": [optimizer.state_dict()],
        "scheduler_states": [cfg["scheduler"].state_dict() for cfg in scheduler_cfgs],
        "best_metric": best_metric,
        "config": config_dict,
    }

    if ckpt_cfg["save_last"]:
        last_path = ckpt_dir / "last.ckpt"
        torch.save(ckpt_payload, last_path)

    if metric_value is None or int(ckpt_cfg["save_top_k"]) == 0:
        return

    filename = _format_checkpoint_name(str(ckpt_cfg["filename"]), epoch, metric_name, metric_value)
    if not filename.endswith(".ckpt"):
        filename = f"{filename}.ckpt"
    path = ckpt_dir / filename
    torch.save(ckpt_payload, path)

    best_k.append((metric_value, path))
    mode = str(ckpt_cfg["mode"])
    reverse = mode == "max"
    best_k.sort(key=lambda x: x[0], reverse=reverse)

    save_top_k = int(ckpt_cfg["save_top_k"])
    if save_top_k > 0:
        while len(best_k) > save_top_k:
            _, to_delete = best_k.pop()
            if to_delete.exists():
                to_delete.unlink()


def fit(
    *,
    diffusion_module: DiffusionModule[BatchedData],
    datamodule: Any,
    trainer_cfg: DictConfig,
    native_cfg: DictConfig,
    config_dict: dict[str, Any],
    ckpt_path: str | None,
    optimizer_partial: OptimizerPartial | None,
    scheduler_partials: list[dict[str, Any]] | list[dict[str, SchedulerPartial]] | None,
) -> None:
    distributed = _is_distributed_env()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None

    if distributed and not dist.is_initialized():
        backend = native_cfg.get("distributed_backend", "nccl")
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    device = _resolve_device(local_rank)
    diffusion_module = diffusion_module.to(device)
    model = diffusion_module

    if native_cfg.get("set_property_scalers", True):
        compute_property_scalers(datamodule=datamodule, property_embeddings=model.model.property_embeddings)
        if hasattr(model.model, "property_embeddings_adapt"):
            compute_property_scalers(
                datamodule=datamodule,
                property_embeddings=model.model.property_embeddings_adapt,
            )

    if distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=native_cfg.get("find_unused_parameters", True),
        )

    optimizer, scheduler_cfgs = _parse_optimizers(
        build_optimizers_and_schedulers(
            diffusion_module=diffusion_module,
            optimizer_partial=optimizer_partial,
            scheduler_partials=scheduler_partials,
        )
    )

    train_loader, train_sampler = build_split_dataloader(
        datamodule,
        "train",
        distributed=distributed,
        shuffle=True,
    )
    if train_loader is None:
        raise ValueError("Native DDP requires a train dataloader.")

    val_loader, _ = build_split_dataloader(
        datamodule,
        "val",
        distributed=distributed,
        shuffle=False,
    )

    max_epochs = int(trainer_cfg.max_epochs)
    grad_clip = float(trainer_cfg.get("gradient_clip_val", 0.0))
    check_val_every_n_epoch = int(trainer_cfg.get("check_val_every_n_epoch", 1))
    use_amp = bool(native_cfg.get("use_amp", False))
    amp_dtype_name = native_cfg.get("amp_dtype", "float16")
    amp_dtype = torch.float16 if amp_dtype_name == "float16" else torch.bfloat16
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)
    ckpt_cfg = _extract_checkpoint_cfg(trainer_cfg)

    output_dir = Path(os.getcwd())
    if _is_rank_zero(rank):
        _save_config_yaml(config_dict=config_dict, output_dir=output_dir)

    wandb_run = None
    logger_cfg = _extract_logger_cfg(trainer_cfg)
    if _is_rank_zero(rank) and logger_cfg is not None:
        try:
            import wandb

            wandb_run = wandb.init(
                project=logger_cfg.get("project"),
                job_type=logger_cfg.get("job_type"),
                config=config_dict,
            )
        except Exception as e:
            logger.warning("WandB init failed, continuing without WandB logging: %s", e)

    best_val = float("inf")
    start_epoch = 0
    if ckpt_path is not None:
        start_epoch, loaded_best = _load_checkpoint(
            ckpt_path,
            diffusion_module=diffusion_module,
            optimizer=optimizer,
            scheduler_cfgs=scheduler_cfgs,
        )
        if loaded_best is not None:
            best_val = float(loaded_best)
        if _is_rank_zero(rank):
            logger.info("Resumed native training from %s at epoch=%s", ckpt_path, start_epoch)

    best_k: list[tuple[float, Path]] = []

    for epoch in range(start_epoch, max_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = 0.0
        train_steps = 0
        for step_idx, batch in enumerate(train_loader):
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, _metrics = calc_loss(model.module, batch) if distributed else calc_loss(model, batch)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                optimizer.step()

            _step_schedulers(scheduler_cfgs, when="step")
            reduced_loss = _mean_reduce(loss.detach(), distributed)
            train_loss_sum += float(reduced_loss.item())
            train_steps += 1

            if _is_rank_zero(rank) and step_idx % int(native_cfg.get("log_every_n_steps", 50)) == 0:
                logger.info(
                    "epoch=%s step=%s loss_train=%.6f",
                    epoch,
                    step_idx,
                    float(reduced_loss.item()),
                )
                if wandb_run is not None:
                    wandb_run.log({"loss_train_step": float(reduced_loss.item()), "epoch": epoch})

        avg_train = train_loss_sum / max(train_steps, 1)
        val_loss = None

        if val_loader is not None and ((epoch + 1) % check_val_every_n_epoch == 0):
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = _to_device(batch, device)
                    loss, _metrics = calc_loss(model.module, batch) if distributed else calc_loss(model, batch)
                    reduced_loss = _mean_reduce(loss.detach(), distributed)
                    val_loss_sum += float(reduced_loss.item())
                    val_steps += 1
            val_loss = val_loss_sum / max(val_steps, 1)

        _step_schedulers(scheduler_cfgs, when="epoch", val_loss=val_loss)

        if _is_rank_zero(rank):
            logger.info(
                "epoch=%s loss_train=%.6f%s",
                epoch,
                avg_train,
                "" if val_loss is None else f" loss_val={val_loss:.6f}",
            )
            if wandb_run is not None:
                metrics = {"loss_train": avg_train, "epoch": epoch}
                if val_loss is not None:
                    metrics["loss_val"] = val_loss
                wandb_run.log(metrics)

            every_n_epochs = int(ckpt_cfg["every_n_epochs"])
            should_checkpoint = (epoch + 1) % every_n_epochs == 0
            if val_loss is not None:
                if str(ckpt_cfg["mode"]) == "min":
                    best_val = min(best_val, val_loss)
                else:
                    best_val = max(best_val, val_loss)

            if should_checkpoint:
                _save_checkpoint(
                    output_dir=output_dir,
                    model=model.module if distributed else model,
                    optimizer=optimizer,
                    epoch=epoch,
                    val_loss=val_loss,
                    config_dict=config_dict,
                    scheduler_cfgs=scheduler_cfgs,
                    ckpt_cfg=ckpt_cfg,
                    best_k=best_k,
                    best_metric=best_val,
                )

        if distributed:
            dist.barrier()

    if distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if wandb_run is not None:
        wandb_run.finish()
