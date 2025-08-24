import json
import random
from pathlib import Path
from typing import List, Tuple
from sqlmodel import select
from ..db import get_session
from ..models import Story, Image
from ..config import settings
from .llm import LocalLLM
from .sd import ImageService
from .thumb import make_thumbnail
from .utils import slugify

def _pick_main(stories: List[Story]) -> Story:
    # naive heuristic: pick first or highest image width if available
    return stories[0]

def generate_frontpage(n: int | None = None) -> Tuple[Story, List[Story]]:
    n = n or settings.app_frontpage_story_count

    llm = LocalLLM(
        model_path=settings.llm_model_path,
        ctx_size=settings.llm_ctx_size,
        gpu_layers=settings.llm_gpu_layers,
        seed=settings.llm_seed,
        temperature=settings.llm_temperature,
    )

    image_out = Path(settings.app_data_dir)
    main_w, main_h = settings.main_image_size()
    sd = ImageService(out_dir=image_out, size=(main_w, main_h))

    # 1) Headlines
    heads = llm.generate_headlines(n)

    created: List[Story] = []
    with get_session() as s:
        for h in heads:
            expanded = llm.expand_story(h["title"], h["summary"], h["keywords"])
            slug = slugify(expanded["title"])
            story = Story(
                slug=slug,
                title=expanded["title"],
                summary=expanded["summary"],
                body_md=expanded["body_md"],
                status="draft",
                topic_tags=json.dumps(expanded["tags"]),
                seed_text=h.get("seed_text",""),
            )
            s.add(story)
            s.commit()
            s.refresh(story)

            # 2) Choose an image prompt and render images
            #    (Merge style_notes into the positive prompt for convenience)
            ip = expanded["image_prompts"][0] if expanded.get("image_prompts") else {
                "positive": expanded["title"],
                "negative": settings.sd_negative,
                "style_notes": ""
            }
            positive = (ip.get("positive","").strip() + ", " + ip.get("style_notes","").strip(", ").strip()).strip(", ")
            negative = ip.get("negative","").strip() or settings.sd_negative
            seeds = [random.randint(1, 10_000_000) for _ in range(2)]
            variants = sd.render_variants(
                story_title=expanded["title"],
                positive_prompt=positive,
                negative_prompt=negative,
                seeds=seeds,
            )

            # 3) Thumbnail
            src = Path("app") / v["path"]  # stored relative to app/
            thumb_path = src.parent / f"thumb_{Path(v['path']).stem}.jpg"
            make_thumbnail(src, thumb_path, settings.thumb_size())

            story.primary_image_id = primary_image_id
            story.status = "published"
            s.add(story)
            s.commit()
            s.refresh(story)
            created.append(story)

        main_story = _pick_main(created)
        # Ensure main is published (already is), others are too
        return main_story, created

def get_published_frontpage(limit: int = 12) -> List[Story]:
    with get_session() as s:
        stories = s.exec(
            select(Story).where(Story.status == "published").order_by(Story.created_at.desc()).limit(limit)
        ).all()
        return stories

def get_story_by_slug(slug: str) -> Story | None:
    with get_session() as s:
        return s.exec(select(Story).where(Story.slug == slug)).first()

def get_story_image(story: Story) -> Image | None:
    if not story.primary_image_id:
        return None
    with get_session() as s:
        return s.get(Image, story.primary_image_id)
