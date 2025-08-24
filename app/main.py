from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from .db import create_db_and_tables, get_session
from .models import Story, Image
from .services.orchestrator import generate_frontpage, get_published_frontpage, get_story_by_slug, get_story_image
from .config import settings

app = FastAPI(title="NewsGen")

# Static files (CSS, images)
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.get("/")
def home(request: Request):
    stories = get_published_frontpage(limit=12)
    main_story = stories[0] if stories else None
    main_image = get_story_image(main_story) if main_story else None

    # build sidebar models with thumbs if present
    sidebar = []
    for s in stories[1:]:
        thumb_path = None
        if s.primary_image_id:
            with get_session() as ses:
                img = ses.get(Image, s.primary_image_id)
                if img:
                    # thumbnails saved as same folder + thumb_cover_*.jpg
                    p = Path("app") / img.path
                    tp = p.parent / f"thumb_{Path(img.path).stem}.jpg"
                    if tp.exists():
                        thumb_path = str(tp).replace("app/static/", "")
        s._thumb_path = f"{thumb_path}" if thumb_path else None  # attach for template
        sidebar.append(s)

    return templates.TemplateResponse(
        "index.html.j2",
        {"request": request, "title": "NewsGen", "main_story": main_story, "main_image": main_image, "sidebar": sidebar}
    )

@app.get("/story/{slug}")
def story_detail(slug: str, request: Request):
    story = get_story_by_slug(slug)
    if not story:
        return RedirectResponse(url="/", status_code=302)
    image = get_story_image(story)
    return templates.TemplateResponse(
        "story.html.j2",
        {"request": request, "title": story.title, "story": story, "image": image}
    )

@app.get("/generate")
def generate():
    # synchronous for simplicity; for long jobs consider BackgroundTasks / APScheduler
    generate_frontpage(n=settings.app_frontpage_story_count)
    return RedirectResponse(url="/", status_code=302)
