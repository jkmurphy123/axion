import random
from typing import List, Dict
from .utils import now_utc_str

class LocalLLM:
    """
    Placeholder LLM service. Replace internals with llama-cpp-python calls.
    """

    def __init__(self, model_path: str, ctx_size: int, gpu_layers: int, seed: int, temperature: float):
        self.model_path = model_path
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.seed = seed
        self.temperature = temperature
        random.seed(seed)

    def generate_headlines(self, n: int = 6) -> List[Dict]:
        pool = [
            ("Mysterious Aurora Over Midwest Skies", "Residents report shimmering bands of color at noon; scientists puzzled.", ["science","weather","mystery","atmosphere"]),
            ("City Council Approves Underground Garden", "Abandoned subway becomes a mile-long hydroponic park.", ["city","urban","green","transport","architecture"]),
            ("Local Cat Wins Chess Tournament", "Tabby named Pixel bests human opponents in blitz rounds.", ["quirky","pets","games","ai?","community"]),
            ("Start-up Sells Bottled Silence", "Soundproof capsules promise ‘mental reset’ in 60 seconds.", ["business","wellness","tech","consumer"]),
            ("Town Library Discovers Hidden Wing", "Renovation reveals sealed corridor of pre-war diaries.", ["history","culture","archives"]),
            ("High School Launches Student Weather Satellites", "Teens assemble cube-sats that tweet forecasts every orbit.", ["education","space","tech","community"]),
        ]
        random.shuffle(pool)
        out = []
        for i in range(n):
            title, summary, kw = pool[i % len(pool)]
            out.append({
                "title": title,
                "summary": summary,
                "keywords": kw,
                "seed_text": f"{title} :: {now_utc_str()}"
            })
        return out

    def expand_story(self, headline: str, summary: str, keywords: List[str]) -> Dict:
        body = (
            f"**{headline}**\n\n"
            f"{summary}\n\n"
            "The scene unfolded earlier today as locals paused and took note. "
            "Eyewitness accounts vary, but the shared sense of surprise is unmistakable. "
            "Officials say they will provide an update later this week while researchers gather more data.\n\n"
            "Meanwhile, residents have begun to organize community discussions and ad-hoc clubs inspired by the phenomenon. "
            "Small businesses are already leaning in, merchandising playful nods to the moment.\n\n"
            "While some observers urge caution, others celebrate the city’s resilience and curiosity. "
            "What happens next may depend on how the community chooses to respond."
        )
        tags = list(dict.fromkeys(keywords + ["local", "feature"]))[:8]
        image_prompts = [
            {
                "positive": f"{headline}, documentary photo, golden hour, 35mm, shallow depth of field",
                "negative": "lowres, blurry, watermark, text, logo",
                "style_notes": "natural color, news wire style, plausible realism"
            },
            {
                "positive": f"{headline}, wide shot, people reacting, streetscape, candid, overcast",
                "negative": "lowres, cartoonish, extra fingers, artifacts",
                "style_notes": "photojournalism, muted tones"
            },
            {
                "positive": f"{headline}, close-up detail, textures, ambient light",
                "negative": "lowres, text, watermark, posterized",
                "style_notes": "editorial macro detail"
            }
        ]
        return {
            "title": headline,
            "summary": summary,
            "body_md": body,
            "tags": tags,
            "image_prompts": image_prompts
        }
