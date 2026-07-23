import argparse
import csv
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from hand_restoration.conditions import ConditionConfig
from hand_restoration.config import load_json_config
from hand_restoration.diffusion import ControlNetHandRestorer, DiffusionConfig
from hand_restoration.hot3d_dataset import Hot3DSingleFrameDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_batch_metadata(batch: dict, key: str) -> str:
    value = batch.get("metadata", {}).get(key, "")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ControlNet on aligned HOT3D MANO restoration samples.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None, help="ControlNet .pt checkpoint from this script.")
    args = parser.parse_args()
    try:
        from accelerate import Accelerator
        from diffusers.optimization import get_scheduler
    except ImportError as exc:
        raise RuntimeError("Install accelerate and diffusers in glove-hot3d before training.") from exc

    config = load_json_config(args.config)
    set_seed(config.get("seed", 0))
    train_cfg = config["training"]
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        mixed_precision=train_cfg.get("mixed_precision", "no"),
        log_with="wandb" if train_cfg.get("use_wandb", False) else None,
    )
    data = config["data"]
    dataset = Hot3DSingleFrameDataset(
        clip_tars=data["clip_tars"], mano_model_dir=data["mano_model_dir"], camera_id=data.get("camera_id", "1201-2"),
        hands=data.get("hands", "right"), output_size=data.get("output_size", 512), grayscale=data.get("grayscale", True), frame_start=data.get("frame_start", 0), frame_stride=data.get("frame_stride", 1),
        max_frames_per_clip=data.get("max_frames_per_clip"), condition=ConditionConfig(**config.get("condition", {})), include_numpy=False,
    )
    limit = train_cfg.get("max_train_samples")
    if limit:
        dataset = Subset(dataset, list(range(min(limit, len(dataset)))))
    loader = DataLoader(dataset, batch_size=train_cfg.get("batch_size", 1), shuffle=True, num_workers=train_cfg.get("num_workers", 0))
    device = accelerator.device
    restorer = ControlNetHandRestorer(DiffusionConfig(**config.get("model", {})), device=device)
    restorer.vae.to(device)
    restorer.text_encoder.to(device)
    restorer.unet.to(device)
    if args.resume:
        restorer.load_controlnet(args.resume)
    resume_step = 0
    if args.resume:
        match = re.search(r"controlnet_step(\d+)$", args.resume.stem)
        if match:
            resume_step = int(match.group(1))
    optimizer = torch.optim.AdamW(restorer.trainable_parameters, lr=train_cfg.get("learning_rate", 1e-5), betas=(0.9, 0.999), weight_decay=train_cfg.get("weight_decay", 1e-2))
    steps = train_cfg.get("max_train_steps", 1000)
    scheduler = get_scheduler(train_cfg.get("lr_scheduler", "constant"), optimizer=optimizer, num_training_steps=steps * accelerator.num_processes, num_warmup_steps=train_cfg.get("warmup_steps", 0))
    restorer.controlnet, optimizer, loader, scheduler = accelerator.prepare(restorer.controlnet, optimizer, loader, scheduler)
    restorer.controlnet.train()
    output = Path(train_cfg.get("output_dir", "outputs/hand_restoration/train"))
    output.mkdir(parents=True, exist_ok=True)
    log_file = None
    log_writer = None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if accelerator.is_main_process:
        log_path = output / "training_log.csv"
        write_header = not log_path.exists() or log_path.stat().st_size == 0
        log_file = log_path.open("a", newline="", encoding="utf-8")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "run_id", "resume_from", "run_step", "total_step", "epoch", "batch_in_epoch", "run_samples_seen",
                "sequence_id", "frame_id",
                "loss", "loss_ema", "learning_rate", "grad_norm",
                "step_seconds", "elapsed_seconds", "cuda_allocated_mb", "cuda_reserved_mb",
            ],
        )
        if write_header:
            log_writer.writeheader()
            log_file.flush()

    global_step = 0
    epoch = 0
    loss_ema = None
    ema_decay = float(train_cfg.get("loss_ema_decay", 0.98))
    csv_log_every = max(1, int(train_cfg.get("csv_log_every", 1)))
    run_start = time.perf_counter()
    previous_step_end = run_start
    try:
        while global_step < steps:
            for batch_in_epoch, batch in enumerate(loader):
                grad_norm_value = float("nan")
                with accelerator.accumulate(restorer.controlnet):
                    loss = restorer.training_loss(batch["target_rgb"], batch["condition_rgb"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite diffusion loss at step {global_step}: {loss.item()}")
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(restorer.controlnet.parameters(), train_cfg.get("max_grad_norm", 1.0))
                        grad_norm_value = float(grad_norm.detach().float().item())
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                loss_value = float(loss.detach().float().item())
                loss_ema = loss_value if loss_ema is None else ema_decay * loss_ema + (1.0 - ema_decay) * loss_value
                now = time.perf_counter()
                step_seconds = now - previous_step_end
                previous_step_end = now

                if accelerator.is_main_process and global_step % csv_log_every == 0:
                    total_step = resume_step + global_step
                    allocated_mb = 0.0
                    reserved_mb = 0.0
                    if torch.cuda.is_available():
                        allocated_mb = torch.cuda.memory_allocated(device) / (1024.0 ** 2)
                        reserved_mb = torch.cuda.memory_reserved(device) / (1024.0 ** 2)
                    log_writer.writerow({
                        "run_id": run_id,
                        "resume_from": str(args.resume or ""),
                        "run_step": global_step,
                        "total_step": total_step,
                        "epoch": epoch,
                        "batch_in_epoch": batch_in_epoch,
                        "run_samples_seen": global_step * train_cfg.get("batch_size", 1) * accelerator.num_processes,
                        "sequence_id": format_batch_metadata(batch, "sequence_id"),
                        "frame_id": format_batch_metadata(batch, "frame_id"),
                        "loss": f"{loss_value:.9g}",
                        "loss_ema": f"{loss_ema:.9g}",
                        "learning_rate": f"{optimizer.param_groups[0]['lr']:.9g}",
                        "grad_norm": f"{grad_norm_value:.9g}",
                        "step_seconds": f"{step_seconds:.6f}",
                        "elapsed_seconds": f"{now - run_start:.6f}",
                        "cuda_allocated_mb": f"{allocated_mb:.3f}",
                        "cuda_reserved_mb": f"{reserved_mb:.3f}",
                    })
                    log_file.flush()

                if accelerator.is_main_process and global_step % train_cfg.get("log_every", 10) == 0:
                    print(f"step={resume_step + global_step:05d} loss={loss_value:.6f} loss_ema={loss_ema:.6f}")
                checkpoint_due = (resume_step + global_step) % train_cfg.get("checkpoint_every", 250) == 0
                if checkpoint_due:
                    accelerator.wait_for_everyone()
                if accelerator.is_main_process and checkpoint_due:
                    total_step = resume_step + global_step
                    checkpoint_path = output / f"controlnet_step{total_step:06d}.pt"
                    torch.save({"config": restorer.config.__dict__, "global_step": total_step, "state_dict": accelerator.unwrap_model(restorer.controlnet).state_dict()}, checkpoint_path)
                if global_step >= steps:
                    break
            epoch += 1
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_path = output / "controlnet_final.pt"
            torch.save({"config": restorer.config.__dict__, "global_step": resume_step + global_step, "state_dict": accelerator.unwrap_model(restorer.controlnet).state_dict()}, checkpoint_path)
            print(f"Saved {output / 'controlnet_final.pt'}")
            print(f"Saved training log to {output / 'training_log.csv'}")
    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
