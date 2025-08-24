from __future__ import annotations

from datetime import datetime
from typing import Optional, List
from sqlmodel import SQLModel, Field, Relationship


class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    # FK to Story (one-to-many side)
    story_id: Optional[int] = Field(default=None, foreign_key="story.id", index=True)

    path: str
    width: int
    height: int
    model_name: Optional[str] = ""
    sampler: Optional[str] = ""
    steps: Optional[int] = 0
    cfg_scale: Optional[float] = 0.0
    seed: Optional[int] = 0
    positive_prompt: Optional[str] = ""
    negative_prompt: Optional[str] = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Disambiguate: join via Image.story_id -> Story.id
    story: Optional["Story"] = Relationship(
        back_populates="images",
        sa_relationship_kwargs={
            "primaryjoin": "Image.story_id==Story.id",
        },
    )


class Story(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    slug: str = Field(index=True, unique=True)
    title: str
    summary: str = ""
    body_md: str = ""
    status: str = Field(default="draft")  # draft|published|error
    topic_tags: Optional[str] = Field(default="[]")  # JSON string for simplicity
    seed_text: Optional[str] = None

    # FK back to Image for the featured/primary image
    primary_image_id: Optional[int] = Field(default=None, foreign_key="image.id")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Disambiguate: join via Image.story_id -> Story.id
    images: List[Image] = Relationship(
        back_populates="story",
        sa_relationship_kwargs={
            "primaryjoin": "Story.id==Image.story_id",
            "cascade": "all, delete-orphan",
        },
    )

    # Optional: convenient relationship for the featured image
    primary_image: Optional[Image] = Relationship(
        sa_relationship_kwargs={
            "primaryjoin": "Story.primary_image_id==Image.id",
            "uselist": False,
        }
    )
