from pathlib import Path
from typing import Tuple, List
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    app_db_url: str = "sqlite:///newsgen.db"
    app_data_dir: str = "app/static/images"
    app_thumb_size: str = "640,360"
    app_main_image_size: str = "1280,720"
    app_frontpage_story_count: int = 6

    llm_model_path: str = "/models/llm/your.gguf"
    llm_ctx_size: int = 4096
    llm_gpu_layers: int = 0
    llm_seed: int = 1234
    llm_temperature: float = 0.8

    sd_device: str = "cuda"
    sd_width: int = 1280
    sd_height: int = 720
    sd_steps: int = 28
    sd_cfg: float = 6.5
    sd_negative: str = "lowres, blurry, jpeg artifacts, watermark"

    @field_validator("app_thumb_size", "app_main_image_size")
    @classmethod
    def _to_tuple_str(cls, v: str) -> str:
        # keep as "w,h" and parse later where needed
        return v

    def thumb_size(self) -> Tuple[int, int]:
        w, h = self.app_thumb_size.split(",")
        return int(w), int(h)

    def main_image_size(self) -> Tuple[int, int]:
        w, h = self.app_main_image_size.split(",")
        return int(w), int(h)

    def data_dir_path(self) -> Path:
        p = Path(self.app_data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

settings = Settings()
