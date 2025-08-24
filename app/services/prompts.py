HEADLINES_SYSTEM = "You are a newsroom editor producing punchy, plausible fictional headlines."
HEADLINES_USER = """Create {n} fictional news headlines with 1–2 sentence summaries and 5–10 keywords each.
Avoid real people and trademarks. Output JSON list."""

EXPAND_SYSTEM = "You are a senior reporter. Write engaging copy in AP style."
EXPAND_USER = """Expand this headline into a 3–6 paragraph article.
Include a 2-sentence summary, 5–10 tags, and 3 image prompts (positive/negative).
HEADLINE: "{headline}"
SUMMARY: "{summary}"
KEYWORDS: {keywords_json}
Output JSON fields: title, summary, body_md, tags, image_prompts: [{positive, negative, style_notes}]
"""
