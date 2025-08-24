import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import os

from PIL import Image, ImageDraw, ImageFont

from .utils import slugify
from ..config import settings

# --------- Placeholder (kept for fallback) ---------
def _placeholder_img(size: Tuple[int, int], title: str, seed: int) -> Image.Image:
    w, h = size
    random.seed(seed)
    a = (random.randint(60,120), random.randint(60,120), random.randint(60,120))
    b = (random.randint(120,200), random.randint(120,200), random.randint(120,200))
    img = Image.new("RGB", (w, h), color=a)
    for y in range(h):
        ratio = y / max(1, h-1)
        r = int(a[0]*(1-ratio) + b[0]*ratio)
        g = int(a[1]*(1-ratio) + b[1]*ratio)
        bl = int(a[2]*(1-ratio) + b[2]*ratio)
        for x in range(w):
            img.putpixel((x,y), (r,g,bl))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=int(h*0.05))
    except:
        font = ImageFont.load_default()
    text = title[:80]
    tw, th = draw.textbbox((0,0), text, font=font)[2:]
    draw.rectangle([(0, h-th-20), (w, h)], fill=(0,0,0,140))
    draw.text(((w-tw)//2, h-th-10), text, fill=(255,255,255), font=font)
    return img


class _DiffusersBackend:
    """
    Lazily loads a diffusers pipeline from a single-file SD checkpoint in
    settings.sd_model_dir / settings.sd_model_file (auto-picked if file empty).
    Supports SD 1.5 and SDXL.
    """
    def __init__(self, size: Tuple[int,int]):
        self.size = size
        self.pipe = None
        self.model_path = None
        self.variant = None  # "sdxl" | "sd15"
        self._try_init()

    def _pick_model(self, root: Path) -> Optional[Path]:
        if settings.sd_model_file:
            p = root / settings.sd_model_file
            return p if p.exists() else None
        # Auto-pick: prefer SDXL, then SD15
        cands = list(root.glob("**/*.safetensors")) + list(root.glob("**/*.ckpt"))
        if not cands:
            return None
        # crude heuristic: prefer files with "xl" in the name
        sdxl = [p for p in cands if "xl" in p.name.lower()]
        return sdxl[0] if sdxl else cands[0]

    def _try_init(self):
        try:
            import torch
            from diffusers import (
                StableDiffusionPipeline,
                StableDiffusionXLPipeline,
                DPMSolverMultistepScheduler,
                EulerAncestralDiscreteScheduler,
            )
        except Exception:
            return

        model_dir = Path(settings.sd_model_dir)
        model_path = self._pick_model(model_dir)
        if not model_path:
            print("[SD] No model file found in", model_dir)
            return

        # Device & dtype
        device = settings.sd_device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        # IMPORTANT: fp16 only on CUDA. CPU requires fp32.
        dtype = torch.float16 if (settings.sd_torch_dtype.lower() in ("fp16","float16") and device == "cuda") else torch.float32

        name = model_path.name.lower()
        is_xl = "xl" in name

        # *** Strongly prefer SD1.5 on Jetson unless explicitly set ***
        # If auto-picked XL but you didn't set SD_MODEL_FILE, fall back to SD1.5 if present.
        if is_xl and not settings.sd_model_file:
            # look for any non-XL as a safer default
            non_xl = [p for p in (list(model_dir.glob("**/*.safetensors")) + list(model_dir.glob("**/*.ckpt"))) if "xl" not in p.name.lower()]
            if non_xl:
                model_path = non_xl[0]
                name = model_path.name.lower()
                is_xl = False

        try:
            Pipe = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline
            pipe = Pipe.from_single_file(
                str(model_path),
                torch_dtype=dtype,
                use_safetensors=model_path.suffix.lower()==".safetensors",
            )

            # Scheduler
            scheduler_name = settings.sd_scheduler.lower()
            if scheduler_name.startswith("dpm"):
                from diffusers import DPMSolverMultistepScheduler
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            elif scheduler_name.startswith("euler"):
                from diffusers import EulerAncestralDiscreteScheduler
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

            # PROBLEMATIC ON JETSON: cpu offload often stalls.
            # -> Keep the whole pipe on a single device.
            if device == "cuda":
                pipe.to("cuda")
            else:
                pipe.to("cpu")

            # Attention slicing can help VRAM, but turn it OFF first while debugging stalls.
            if hasattr(pipe, "disable_attention_slicing"):
                pipe.disable_attention_slicing()
            if hasattr(pipe, "set_progress_bar_config"):
                pipe.set_progress_bar_config(disable=True)

            # Optional, sometimes helps allocator on Jetson
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            self.pipe = pipe
            self.model_path = model_path
            self.variant = "sdxl" if is_xl else "sd15"
            print(f"[SD] Loaded {self.variant.upper()} model: {model_path.name} on {device} dtype={dtype}")
        except Exception as e:
            print("[SD] Failed to init:", e)
            self.pipe = None

    # inside _DiffusersBackend.render_one(self, positive_prompt, negative_prompt, seed):
    def render_one(self, positive_prompt: str, negative_prompt: str, seed: int) -> Optional[Image.Image]:
        if self.pipe is None:
            return None
        try:
            import torch
            # Clamp sizes to safe defaults, especially for SDXL
            w, h = self.size
            if self.variant == "sdxl":
                w = min(w, 1024)
                h = min(h, 1024)
            else:
                # SD1.5: start conservative; you can raise later
                w = min(w, 768)
                h = min(h, 768)

            generator = torch.Generator(device=str(self.pipe.device)).manual_seed(seed)
            return self.pipe(
                prompt=positive_prompt,
                negative_prompt=negative_prompt or settings.sd_negative,
                num_inference_steps=settings.sd_steps,
                guidance_scale=settings.sd_cfg,
                width=w, height=h,
                generator=generator,
            ).images[0]
        except Exception as e:
            print("[SD] Generation error:", e)
            return None


class ImageService:
    """
    Unified image service:
      - Try diffusers backend with your local SD model
      - If unavailable, fall back to placeholder generator
    Paths returned are relative to the app/ folder, same as before.
    """
    def __init__(self, out_dir: Path, size: Tuple[int, int]):
        self.out_dir = out_dir
        self.size = size
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._backend = _DiffusersBackend(size=size)

    def _save_img(self, img: Image.Image, story_slug: str, seed: int) -> str:
        story_dir = self.out_dir / story_slug
        story_dir.mkdir(parents=True, exist_ok=True)
        path = story_dir / f"cover_{seed}.jpg"
        img.save(path, quality=92, subsampling=1)
        # Return path relative to app/
        return str(path.relative_to(self.out_dir.parent.parent))

    def render_variants(
        self,
        story_title: str,
        positive_prompt: str,
        negative_prompt: str,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict]:
        slug = slugify(story_title)
        results: List[Dict] = []
        seeds = seeds or [random.randint(1, 10_000_000) for _ in range(2)]

        for seed in seeds:
            img = None
            # Try real SD first
            if self._backend.pipe is not None:
                img = self._backend.render_one(positive_prompt, negative_prompt, seed)

            # Fallback placeholder
            if img is None:
                img = _placeholder_img(self.size, story_title, seed)

            rel = self._save_img(img, slug, seed)
            results.append({
                "path": rel,
                "width": img.width,
                "height": img.height,
                "model_name": (self._backend.model_path.name if self._backend.model_path else "placeholder"),
                "sampler": settings.sd_scheduler if self._backend.pipe else "placeholder",
                "steps": settings.sd_steps if self._backend.pipe else 0,
                "cfg_scale": settings.sd_cfg if self._backend.pipe else 0.0,
                "seed": seed,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt or settings.sd_negative,
            })
        return results
