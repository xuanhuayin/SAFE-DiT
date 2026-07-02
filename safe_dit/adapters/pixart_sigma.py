"""SAFE-DiT generation adapter for PixArt-Sigma diffusers pipelines.

This adapter is intentionally self-contained and uses public Hugging Face model
IDs. It does not depend on local research directories. The exact paper numbers
use the paper's full experimental stack, while this public adapter exposes the
same SAFE-DiT mechanisms in a reproducible diffusers script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch

from ..regions import build_sensitivity_partition
from ..scheduler import SAFEConfig, first_order_extrapolate, sensitivity_weighted_cfg_image, should_compute_dense_step


@dataclass(frozen=True)
class PixArtModelSpec:
    transformer_id: str = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
    pipeline_id: str = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    revision: Optional[str] = None


class PixArtSAFEGenerator:
    """Dense and SAFE-DiT generation for PixArt-Sigma."""

    def __init__(
        self,
        spec: Optional[PixArtModelSpec] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dir: Optional[str] = None,
    ) -> None:
        try:
            from diffusers import PixArtSigmaPipeline, Transformer2DModel
        except ImportError as exc:
            raise ImportError(
                "PixArtSAFEGenerator requires diffusers. Install the generation dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        self.spec = spec or PixArtModelSpec()
        self.device = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        transformer = Transformer2DModel.from_pretrained(
            self.spec.transformer_id,
            subfolder="transformer",
            torch_dtype=dtype,
            cache_dir=cache_dir,
            revision=self.spec.revision,
        )
        self.pipe = PixArtSigmaPipeline.from_pretrained(
            self.spec.pipeline_id,
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            revision=self.spec.revision,
        )
        self.pipe.to(self.device)

    def generate_dense(
        self,
        prompt: str,
        seed: int = 0,
        steps: int = 20,
        guidance_scale: float = 4.5,
        height: int = 1024,
        width: int = 1024,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            use_resolution_binning=False,
        ).images[0]

    @torch.no_grad()
    def generate_safe(
        self,
        prompt: str,
        seed: int = 0,
        steps: int = 20,
        height: int = 1024,
        width: int = 1024,
        cfg: Optional[SAFEConfig] = None,
    ):
        """Generate with public SAFE-DiT scheduling and spatial CFG.

        The public diffusers block API does not expose row-level query execution
        for every backbone, so this adapter uses SAFE-DiT's sensitivity map,
        anchor refresh, first-order state reuse, and SW-CFG at the denoiser
        prediction level.
        """

        cfg = cfg or SAFEConfig()
        pipe = self.pipe
        device = self.device
        dtype = pipe.transformer.dtype
        latent_channels = pipe.transformer.config.in_channels
        vae_scale = pipe.vae_scale_factor

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = pipe.encode_prompt(
            prompt,
            do_classifier_free_guidance=True,
            device=device,
            num_images_per_prompt=1,
        )
        prompt_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
        prompt_mask = torch.cat([neg_mask, prompt_mask], dim=0)

        generator = torch.Generator(device=device).manual_seed(seed)
        latents = torch.randn(
            (1, latent_channels, height // vae_scale, width // vae_scale),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        latents = latents * pipe.scheduler.init_noise_sigma
        pipe.scheduler.set_timesteps(steps, device=device)
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        sensitivity_map = None
        region_map = None
        eps_history = []

        def forward_eps(latent_model_input: torch.Tensor, timestep: torch.Tensor):
            model_input = torch.cat([latent_model_input] * 2)
            model_input = pipe.scheduler.scale_model_input(model_input, timestep)
            current_timestep = timestep
            if not torch.is_tensor(current_timestep):
                current_timestep = torch.tensor([current_timestep], device=device)
            current_timestep = current_timestep.expand(model_input.shape[0])
            raw = pipe.transformer(
                model_input,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_mask,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            unconditional, conditional = raw.chunk(2)
            if pipe.transformer.config.out_channels // 2 == latent_channels:
                unconditional = unconditional.chunk(2, dim=1)[0]
                conditional = conditional.chunk(2, dim=1)[0]
            return unconditional, conditional

        for step_index, timestep in enumerate(pipe.scheduler.timesteps):
            dense_step = should_compute_dense_step(
                step=step_index,
                total_steps=steps,
                mask_step=cfg.mask_step,
                skip_interval=cfg.skip_interval,
                anchor_interval=cfg.anchor_interval,
            )
            if dense_step or len(eps_history) < 2:
                unconditional, conditional = forward_eps(latents, timestep)
                if step_index == cfg.mask_step:
                    sensitivity_map, region_map = build_sensitivity_partition(
                        unconditional,
                        conditional,
                        keep_ratio=cfg.keep_ratio,
                        eps=cfg.eps,
                    )
                if cfg.sw_cfg and region_map is not None:
                    eps, _ = sensitivity_weighted_cfg_image(
                        unconditional,
                        conditional,
                        region_map.to(device=device, dtype=unconditional.dtype),
                        cfg_context=cfg.cfg_context,
                        cfg_sensitive=cfg.cfg_sensitive,
                    )
                else:
                    eps = unconditional + cfg.cfg_scale * (conditional - unconditional)
                eps_history.append(eps)
                eps_history = eps_history[-2:]
            else:
                eps = first_order_extrapolate(tuple(eps_history))

            latents = pipe.scheduler.step(eps.to(latents.dtype), timestep, latents, return_dict=False)[0]

        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type="pil")[0]

    def save(
        self,
        prompt: str,
        output: Union[str, Path],
        mode: str = "safe",
        seed: int = 0,
        steps: int = 20,
        height: int = 1024,
        width: int = 1024,
        cfg: Optional[SAFEConfig] = None,
    ) -> Path:
        if mode == "dense":
            image = self.generate_dense(prompt, seed=seed, steps=steps, height=height, width=width)
        elif mode == "safe":
            image = self.generate_safe(prompt, seed=seed, steps=steps, height=height, width=width, cfg=cfg)
        else:
            raise ValueError("mode must be 'dense' or 'safe'")
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path
