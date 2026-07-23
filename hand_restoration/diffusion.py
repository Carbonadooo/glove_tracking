from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


def _require_diffusers() -> None:
    try:
        import accelerate  # noqa: F401
        import diffusers  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Diffusion dependencies are missing. Install the glove-hot3d additions "
            "from environment.glove-hot3d.yml, then rerun this command."
        ) from exc


@dataclass(frozen=True)
class DiffusionConfig:
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_model_id: str | None = None
    prompt: str = "a realistic egocentric image of a human hand"
    negative_prompt: str = "deformed hand, extra fingers, blurry, cartoon"
    prediction_type: str | None = None


class ControlNetHandRestorer(torch.nn.Module):
    """Frozen SD 1.5 backbone with a trainable ControlNet condition branch."""

    def __init__(self, config: DiffusionConfig, device: str | torch.device = "cpu") -> None:
        super().__init__()
        _require_diffusers()
        from diffusers import ControlNetModel, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
        from transformers import CLIPTextModel, CLIPTokenizer

        self.config = config
        self.device_name = torch.device(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(config.base_model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(config.base_model_id, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(config.base_model_id, subfolder="vae")
        self.unet = UNet2DConditionModel.from_pretrained(config.base_model_id, subfolder="unet")
        self.noise_scheduler = DDPMScheduler.from_pretrained(config.base_model_id, subfolder="scheduler")
        if config.controlnet_model_id:
            self.controlnet = ControlNetModel.from_pretrained(config.controlnet_model_id)
        else:
            self.controlnet = ControlNetModel.from_unet(self.unet)
        for module in (self.vae, self.text_encoder, self.unet):
            module.requires_grad_(False)
            module.eval()
        if config.prediction_type:
            self.noise_scheduler.register_to_config(prediction_type=config.prediction_type)

    @property
    def trainable_parameters(self):
        return self.controlnet.parameters()

    @torch.no_grad()
    def text_embeddings(self, batch_size: int, prompt: str | None = None) -> torch.Tensor:
        tokens = self.tokenizer(
            [prompt or self.config.prompt] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device_name)
        return self.text_encoder(tokens)[0]

    def training_loss(self, target_rgb: torch.Tensor, condition_rgb: torch.Tensor) -> torch.Tensor:
        """Standard latent diffusion epsilon-prediction objective.

        Both inputs must be Bx3xHxW float images in [-1, 1].
        """
        target_rgb = target_rgb.to(self.device_name)
        condition_rgb = condition_rgb.to(self.device_name)
        with torch.no_grad():
            latents = self.vae.encode(target_rgb).latent_dist.sample() * self.vae.config.scaling_factor
            text = self.text_embeddings(target_rgb.shape[0])
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device, dtype=torch.long)
        noisy = self.noise_scheduler.add_noise(latents, noise, timesteps)
        down, mid = self.controlnet(
            noisy,
            timesteps,
            encoder_hidden_states=text,
            controlnet_cond=(condition_rgb + 1.0) / 2.0,
            return_dict=False,
        )
        prediction = self.unet(
            noisy,
            timesteps,
            encoder_hidden_states=text,
            down_block_additional_residuals=down,
            mid_block_additional_residual=mid,
        ).sample
        target = noise
        if self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        return F.mse_loss(prediction.float(), target.float(), reduction="mean")

    def save_controlnet(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": self.config.__dict__, "state_dict": self.controlnet.state_dict()}, path)

    def load_controlnet(self, path: str | Path, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.controlnet.load_state_dict(checkpoint["state_dict"], strict=strict)

    @torch.no_grad()
    def generate(self, condition_rgb: torch.Tensor, steps: int = 30, guidance_scale: float = 5.0, controlnet_scale: float = 1.0, seed: int = 0):
        from diffusers import StableDiffusionControlNetPipeline

        condition = condition_rgb.to(self.device_name)
        pipeline = StableDiffusionControlNetPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            controlnet=self.controlnet,
            scheduler=self.noise_scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        ).to(self.device_name)
        generator = torch.Generator(device=self.device_name).manual_seed(seed)
        from PIL import Image
        image = ((condition[0].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).clip(0, 255).astype("uint8")
        return pipeline(
            prompt=self.config.prompt,
            negative_prompt=self.config.negative_prompt,
            image=Image.fromarray(image),
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
        ).images[0]
