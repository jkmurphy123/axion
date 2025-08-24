from pathlib import Path
from PIL import Image
from typing import Tuple

def make_thumbnail(src_path: Path, dst_path: Path, size: Tuple[int,int]) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src_path).convert("RGB")
    img.thumbnail(size)
    img.save(dst_path, quality=88)
