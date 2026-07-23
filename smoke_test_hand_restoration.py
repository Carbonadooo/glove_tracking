"""Minimal deterministic acceptance test for the single-frame pipeline.

The default test needs only existing HOT3D/MANO dependencies. ``--diffusion``
also downloads/loads the configured public SD model and checks one diffusion
forward pass plus ControlNet checkpoint round-trip.
"""

import argparse
from pathlib import Path

import torch

from hand_restoration.conditions import ConditionConfig
from hand_restoration.config import load_json_config
from hand_restoration.hot3d_dataset import Hot3DSingleFrameDataset
from hand_restoration.visualize import save_debug_grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/hand_restoration/tiny_overfit.json"))
    parser.add_argument("--diffusion", action="store_true", help="Also load SD/ControlNet and test a training loss + checkpoint reload.")
    parser.add_argument("--inference", action="store_true", help="Also run one low-step pretrained ControlNet inference call.")
    args = parser.parse_args()
    config = load_json_config(args.config)
    data = config["data"]
    dataset = Hot3DSingleFrameDataset(clip_tars=data["clip_tars"], mano_model_dir=data["mano_model_dir"], camera_id=data.get("camera_id", "1201-2"), hands=data.get("hands", "right"), output_size=data.get("output_size", 512), grayscale=data.get("grayscale", True), frame_start=data.get("frame_start", 0), max_frames_per_clip=1, condition=ConditionConfig(**config.get("condition", {})))
    sample = dataset[0]
    assert sample["target_rgb"].shape == sample["condition_rgb"].shape == sample["mano_rgb"].shape
    assert sample["target_rgb"].shape[0] == 3 and sample["mano_mask"].shape[0] == sample["edit_mask"].shape[0] == 1
    assert torch.all(sample["target_rgb"] <= 1) and torch.all(sample["target_rgb"] >= -1)
    assert torch.all(sample["mano_mask"] >= 0) and torch.all(sample["mano_mask"] <= 1)
    output = Path("outputs/hand_restoration/smoke")
    save_debug_grid(sample, output / "preprocess.png")
    print("PASS preprocessing: sample, aligned MANO render, condition, masks, shapes, and ranges.")
    if not args.diffusion:
        return
    from hand_restoration.diffusion import ControlNetHandRestorer, DiffusionConfig
    device = "cuda" if torch.cuda.is_available() else "cpu"
    restorer = ControlNetHandRestorer(DiffusionConfig(**config.get("model", {})), device=device)
    restorer.vae.to(device)
    restorer.text_encoder.to(device)
    restorer.unet.to(device)
    restorer.controlnet.to(device)
    loss = restorer.training_loss(sample["target_rgb"].unsqueeze(0), sample["condition_rgb"].unsqueeze(0))
    assert torch.isfinite(loss), "Diffusion loss is non-finite"
    checkpoint = output / "controlnet_roundtrip.pt"
    restorer.save_controlnet(checkpoint)
    restorer.load_controlnet(checkpoint)
    print(f"PASS diffusion forward and checkpoint reload: loss={loss.item():.6f}")
    if args.inference:
        generated = restorer.generate(sample["condition_rgb"].unsqueeze(0), steps=1, guidance_scale=1.0, controlnet_scale=1.0, seed=7)
        generated.save(output / "pretrained_inference.png")
        print("PASS pretrained ControlNet inference call.")


if __name__ == "__main__":
    main()
