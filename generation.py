import os
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import config


_PIPE = None


def load_stable_diffusion():
    global _PIPE

    if _PIPE is not None:
        return _PIPE

    dtype = torch.float16 if config.DEVICE.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        config.STABLE_DIFFUSION_MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(config.DEVICE)

    if config.DEVICE.type == "cuda":
        pipe.enable_attention_slicing()

    _PIPE = pipe
    return _PIPE


def generate_images_from_prompts(prompts, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    pipe = load_stable_diffusion()

    generated_paths = []

    for i, prompt in enumerate(tqdm(prompts, desc="Generating images")):
        result = pipe(
            prompt,
            guidance_scale=config.STABLE_DIFFUSION_GUIDANCE_SCALE,
            num_inference_steps=config.DIFFUSION_STEPS,
        )

        image = result.images[0]

        save_path = os.path.join(output_dir, f"sample_{i:03d}.png")
        image.save(save_path)
        generated_paths.append(save_path)

    return generated_paths