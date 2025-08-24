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
    """Naive heuristic: pick the first created story as the main feature."""
    return stories[0]


def _slugify_unique(session, title: str) -> str:
    """Generate a unique slug by appending -2, -3, ... if needed."""
    base = slugify(title)
    slug = base
    i = 2
    while session.exec(select(Story).where(Story.slug == slug)).first():
        slug = f"{base}-{i}"
        i += 1
    return slug


def generate_frontpage(n: int | None = None) -> Tuple[Story, List[Story]]:
    """
    Generate a front page worth of stories:
      - headlines (LLM)
      - expanded article (LLM)
      - 1–2 images (SD or placeholder)
      - thumbnails
      - publish all
    Returns (main_story, all_created_stories)
    """
    n = n or settings.app_frontpage_story_count

    # LLM + SD backends
    llm = LocalLLM(
        model_path=settings.llm_model_path,
        ctx_size=settings.llm_ctx_size,
        gpu_layers=settings.llm_gpu_layers,
        seed=settings.llm_seed,
        temperature=settings.llm_temperature,
    )

    image_out = Path(settings.app_data_dir)  # e.g. app/static/images
    main_w, main_h = settings.main_image_size()
    sd = ImageService(out_dir=image_out, size=(main_w, main_h))

    # 1) Headlines (over-generate if you want to filter later)
    heads = llm.generate_headlines(n)

    created: List[Story] = []
    with get_session() as s:
        for h in heads:
            expanded = llm.expand_story(h["title"], h["summary"], h["keywords"])

            # 2) Insert Story
            slug = _slugify_unique(s, expanded["title"])
            story = Story(
                slug=slug,
                title=expanded["title"],
                summary=expanded["summary"],
                body_md=expanded["body_md"],
                status="draft",
                topic_tags=json.dumps(expanded.get("tags", [])),
                seed_text=h.get("seed_text", ""),
            )
            s.add(story)
            s.commit()
            s.refresh(story)

            # 3) Choose an image prompt and render images
            ip = expanded["image_prompts"][0] if expanded.get("image_prompts") else {
                "positive": expanded["title"],
                "negative": settings.sd_negative,
                "style_notes": ""
            }
            positive = (ip.get("positive", "").strip() + ", " + ip.get("style_notes", "").strip(", ").strip()).strip(", ")
            negative = ip.get("negative", "").strip() or settings.sd_negative

            seeds = [random.randint(1, 10_000_000) for _ in range(2)]
            variants = sd.render_variants(
                story_title=expanded["title"],
                positive_prompt=positive,
                negative_prompt=negative,
                seeds=seeds,
            )

            # If SD failed silently, ensure at least one placeholder image
            if not variants:
                variants = sd.render_variants(
                    story_title=expanded["title"],
                    positive_prompt=expanded["title"],
                    negative_prompt=settings.sd_negative,
                    seeds=[random.randint(1, 10_000_000)],
                )

            primary_image_id = None
            for v in variants:
                img = Image(
                    story_id=story.id,
                    path=v["path"],
                    width=v["width"],
                    height=v["height"],
                    model_name=v["model_name"],
                    sampler=v["sampler"],
                    steps=v["steps"],
                    cfg_scale=v["cfg_scale"],
                    seed=v["seed"],
                    positive_prompt=v["positive_prompt"],
                    negative_prompt=v["negative_prompt"],
                )
                s.add(img)
                s.commit()
                s.refresh(img)

                if primary_image_id is None:
                    primary_image_id = img.id

                # 4) Thumbnail (do this inside the loop where `v` is in scope)
                src = Path("app") / v["path"]  # returned path is relative to app/
                thumb_path = src.parent / f"thumb_{Path(v['path']).stem}.jpg"
                make_thumbnail(src, thumb_path, settings.thumb_size())

            # 5) Publish story
            story.primary_image_id = primary_image_id
            story.status = "published"
            s.add(story)
            s.commit()
            s.refresh(story)

            created.append(story)

    main_story = _pick_main(created)
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
    if not story or not story.primary_image_id:
        return None
    with get_session() as s:
        return s.get(Image, story.primary_image_id)
