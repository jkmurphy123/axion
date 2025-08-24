import random
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
from .utils import slugify

class ImageService:
    """
    Placeholder SD service: creates simple images with a gradient + text.
    Replace `render_variants` internals with Diffusers or HTTP call to A1111/Comfy.
    """

    def __init__(self, out_dir: Path, size: Tuple[int, int]):
        self.out_dir = out_dir
        self.size = size
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _placeholder_img(self, title: str, seed: int) -> Image.Image:
        w, h = self.size
        random.seed(seed)
        a = (random.randint(60,120), random.randint(60,120), random.randint(60,120))
        b = (random.randint(120,200), random.randint(120,200), random.randint(120,200))
        img = Image.new("RGB", (w, h), color=a)
        # simple vertical gradient
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

    def render_variants(self, story_title: str, n: int = 2) -> List[Dict]:
        slug = slugify(story_title)
        story_dir = self.out_dir / slug
        story_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for i in range(n):
            seed = random.randint(1, 10_000_000)
            img = self._placeholder_img(story_title, seed)
            path = story_dir / f"cover_{seed}.jpg"
            img.save(path, quality=92, subsampling=1)
            results.append({
                "path": str(path.relative_to(self.out_dir.parent.parent)),  # make path relative to app/
                "width": img.width,
                "height": img.height,
                "model_name": "placeholder",
                "sampler": "placeholder",
                "steps": 0, "cfg_scale": 0.0, "seed": seed,
                "positive_prompt": f"{story_title} (placeholder)",
                "negative_prompt": ""
            })
        return results
