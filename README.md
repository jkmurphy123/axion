# axion
A fictional news site generated on-the-fly by local AI

newsgen/
  .gitignore
  README.md
  requirements.txt
  .env.example
  app/
    __init__.py
    main.py
    config.py
    db.py
    models.py
    services/
      __init__.py
      prompts.py
      utils.py
      llm.py
      sd.py
      orchestrator.py
      thumb.py
    templates/
      base.html.j2
      index.html.j2
      story.html.j2
    static/
      css/
        styles.css
      images/
        .gitkeep
  scripts/
    seed_demo.py
  tests/
    .gitkeep

# NewsGen (starter)

Local-news generator scaffold with **FastAPI + Jinja + SQLite**, stubbed **LLM** and **Stable Diffusion** services.
Ready to run: generates placeholder stories + images and renders a news-style homepage.

## Quick start

cd projects
git clone https://github.com/jkmurphy123/axion.git
cd axion
#python -m venv .venv
source llm_env/bin/activate  
pip install -r requirements.txt

# Optional: copy env
cp .env.example .env

# Run
uvicorn app.main:app --reload
# open http://127.0.0.1:8000
