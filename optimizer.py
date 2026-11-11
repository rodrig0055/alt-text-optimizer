"""Alt-text optimizer: write, score against the image, rewrite, repeat.

A vision call writes a description of the photo and first candidates; each rewrite round is also a vision
call (the writer sees the photo plus the scored history). Every candidate is scored against a cached image
embedding, and the loop stops as soon as one reaches the target match. The current alt text competes too,
so a good human-written alt is kept rather than replaced.

Writers: Gemini (default, cheapest), OpenAI, Anthropic.
Scorers: Google gemini-embedding-2 (multimodal: image and text in one vector space) and local SigLIP 2
(free fallback; its scores are not comparable to Google's).
"""

from __future__ import annotations

import config
import base64
import io
import json
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent
DATA = ROOT / "data"
RESULTS = DATA / "results"

DEFAULTS = {
    "writer": "gemini:gemini-2.5-flash-lite",  # cheapest, and tied-best in a 5-writer comparison
    "scorer": "gemini",
    "target": 0.508,  # raw match needed to stop = 80% on the 0-100 scale below
    "max_rounds": 3,  # rewrite rounds after the first call; every round sees the image
    "candidates": 6,
    "calls_if_met": 3,  # writer calls when the current alt already meets the target (first call + 2 rewrites)
}

# Raw gemini-embedding-2 cosine -> a readable 0-100% match: pct = A + B * raw, clipped to 0..100.
# Raw cosine between an image and its best caption tops out around 0.55, so the raw number reads badly.
# The fit maps a near-perfect caption to ~86% and a meaningless one (".") to ~59%, matching the scale of
# public image-SEO tools that use the same embedding model. Refit it if you calibrate against another tool.
SCALE_A, SCALE_B = 19.2, 119.6


def to_pct(raw: float) -> float:
    return max(0.0, min(100.0, SCALE_A + SCALE_B * raw))


def from_pct(pct: float) -> float:
    return (pct - SCALE_A) / SCALE_B


# Products run when no ids are given: the first few, for a quick look.
QUICK_RUN = 5

# USD per 1M tokens (input, output). OpenAI from developers.openai.com model pages; Anthropic from the
# claude-api model table (2026-10-06). Check before budgeting.
PRICES = {
    "openai:gpt-5.4-nano": (0.20, 1.25),
    "openai:gpt-5.4-mini": (0.75, 4.50),
    "openai:gpt-4.1-nano": (0.10, 0.40),
    "anthropic:claude-haiku-5-5": (0.10, 0.50),
    # aistudio.google.com/docs/pricing, read 2026-11-03 (3.8 Flash doubles on 2027-01-01)
    "gemini:gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini:gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini:gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini:gemini-3.8-flash": (0.75, 3.75),
}
# Many CMSs store alt text as varchar(255); keep a margin. Longer candidates are scored but can never win.
MAX_ALT_CHARS = 250

# Request pacing for Google's free tier (requests per minute). Override with WRITER_RPM / EMBED_RPM.
RPM = {"writer": float(os.environ.get("WRITER_RPM", 12)), "embed": float(os.environ.get("EMBED_RPM", 80))}
_last_call: dict[str, float] = {}


class DailyQuotaExhausted(RuntimeError):
    """The API says today's quota is used up; stop the run and resume tomorrow."""


def throttle(kind: str) -> None:
    gap = 60.0 / RPM[kind]
    wait = _last_call.get(kind, 0) + gap - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_call[kind] = time.time()


# gemini-embedding-2: $0.00012 per image, $0.20 per 1M text tokens (~4 chars per token)
EMBED_IMAGE_USD, EMBED_TEXT_USD_PER_CHAR = 0.00012, 0.20 / 1e6 / 4


def env(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    raise RuntimeError(f"{name} is not set (put it in .env, see .env.example)")


def post_json(url: str, body: dict, headers: dict, timeout: int = 120, retries: int = 6) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data, {"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            text = e.read().decode()
            if e.code == 429 and ("PerDay" in text or "per day" in text.lower()):
                raise DailyQuotaExhausted(text[:400]) from None
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                m = re.search(r'"retryDelay":\s*"(\d+)', text)
                time.sleep(int(m.group(1)) + 1 if m else min(60, 5 * 2 ** attempt))
                continue
            raise RuntimeError(f"{url} -> {e.code}: {text[:400]}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt < retries:
                time.sleep(min(60, 5 * 2 ** attempt))
                continue
            raise RuntimeError(f"{url} -> {e}") from None
    raise RuntimeError(f"{url} -> retries exhausted")


# ---------------------------------------------------------------- images

@lru_cache(maxsize=64)
def load_image(path: str) -> Image.Image:
    return Image.open(DATA / path).convert("RGB")


def image_jpeg_b64(path: str, side: int = 512) -> str:
    img = load_image(path).copy()
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------- writers

VISION_PROMPT = """You write English image alt text for {shop}.

Product name: {name}
Product page copy (use it only to name flowers and items correctly, never to add things the photo does not show):
{copy}

Look at the photo and return JSON with:
- "description": a factual inventory of what is visible: each flower type and colour, greenery, wrapping or
  container and its colour, ribbons, cards, chocolates or other items, people, background. Name flowers
  using the product copy where it matches what you see. If unsure of a flower, say so.
- "candidates": {n} different alt texts describing the image for someone who cannot see it. Each one plain
  sentence of at most 250 characters (database limit). What scores well is precision, not length: name the exact things (e.g. "sliced
  strawberries", "rosemary sprigs", "pink spray roses", "kraft paper wrap"), with colours and materials.
  Never hedge ("strawberry-like", "possibly", "some kind of"): if the copy names an item and the photo shows
  it, use the name. Lead with the main subject. No "image of", no marketing words, no brand names unless
  printed in the photo. Vary the level of detail across the candidates."""

REWRITE_PROMPT = """You improve English image alt text for {shop}. Look at the photo again.

Product name: {name}
Earlier description of the photo (correct it if the photo shows otherwise):
{description}

Earlier alt texts and how closely an image-text embedding model matched each to the photo (higher is better):
{history}

Write {n} new alt texts that should match the photo more closely than the best one above. One plain
sentence each, at most 250 characters (database limit). Precision beats length: replace vague words with the exact name of what is
visible (flower species, fruit, foliage, wrapping material, container), keep what the best texts got right,
and drop details that did not help. Never hedge. Only describe what the photo shows. Vary wording, order
and level of detail.
Return JSON: {{"candidates": [...]}}"""


def call_writer(writer: str, prompt: str, image_b64: str | None) -> tuple[dict, dict]:
    provider, model = writer.split(":", 1)
    if provider == "openai":
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}", "detail": "high"}})
        body = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": 4000,
        }
        if model.startswith("gpt-5"):
            body["reasoning_effort"] = "low"
        r = post_json("https://api.openai.com/v1/chat/completions", body, {"authorization": f"Bearer {env('OPENAI_API_KEY')}"})
        text = r["choices"][0]["message"]["content"]
        usage = {"input": r["usage"]["prompt_tokens"], "output": r["usage"]["completion_tokens"]}
    elif provider == "anthropic":
        content = []
        if image_b64:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}})
        content.append({"type": "text", "text": prompt + "\n\nReturn only the JSON object."})
        r = post_json(
            "https://api.anthropic.com/v1/messages",
            {"model": model, "max_tokens": 4000, "output_config": {"effort": "low"}, "messages": [{"role": "user", "content": content}]},
            {"x-api-key": env("ANTHROPIC_API_KEY"), "anthropic-version": "2023-06-01"},
        )
        text = "".join(b.get("text", "") for b in r["content"] if b["type"] == "text")
        text = text[text.find("{") : text.rfind("}") + 1]
        usage = {"input": r["usage"]["input_tokens"], "output": r["usage"]["output_tokens"]}
    elif provider == "gemini":
        parts: list[dict] = [{"text": prompt}]
        if image_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})
        cfg: dict = {
            "responseMimeType": "application/json",
            "responseSchema": {"type": "OBJECT", "properties": {
                "description": {"type": "STRING"}, "candidates": {"type": "ARRAY", "items": {"type": "STRING"}}}},
        }
        if model.startswith("gemini-3"):
            cfg["thinkingConfig"] = {"thinkingLevel": "low"}
        throttle("writer")
        r = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            {"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg},
            {"x-goog-api-key": GeminiScorer().key},
        )
        text = "".join(p.get("text", "") for p in r["candidates"][0]["content"]["parts"] if not p.get("thought"))
        m = r.get("usageMetadata", {})
        usage = {"input": m.get("promptTokenCount", 0), "output": m.get("candidatesTokenCount", 0) + m.get("thoughtsTokenCount", 0)}
    else:
        raise ValueError(f"unknown writer {writer}")
    # Tolerate trailing text after the JSON object (seen with Gemini): parse the first object only.
    text = text.strip()
    return json.JSONDecoder().raw_decode(text[text.find("{"):])[0], usage


# ---------------------------------------------------------------- scorers

class SiglipScorer:
    """Local SigLIP 2. match = the model's own sigmoid image-text match probability."""

    name = "siglip"
    model_id = "google/siglip2-base-patch16-256"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self.proc = AutoProcessor.from_pretrained(self.model_id)
        self._img: dict[str, object] = {}

    def image(self, path: str):
        if path not in self._img:
            with self.torch.no_grad():
                inp = self.proc(images=load_image(path), return_tensors="pt").to(self.device)
                e = self.model.get_image_features(**inp)
                self._img[path] = e / e.norm(dim=-1, keepdim=True)
        return self._img[path]

    def score(self, path: str, texts: list[str]) -> list[dict]:
        torch = self.torch
        with torch.no_grad():
            inp = self.proc(text=[t.lower() for t in texts], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
            t = self.model.get_text_features(**inp)
            t = t / t.norm(dim=-1, keepdim=True)
            cos = (t @ self.image(path).T).squeeze(-1)
            logits = cos * self.model.logit_scale.exp() + self.model.logit_bias
            prob = torch.sigmoid(logits)
        return [{"cos": round(float(c), 4), "match": round(float(p), 4)} for c, p in zip(cos, prob)]


class GeminiScorer:
    """Google gemini-embedding-2, a multimodal embedding model: image and text land in one vector space,
    and the match is their cosine.

    Auth: GOOGLE_GENERATIVE_AI_API_KEY (or GEMINI_API_KEY) if set, else Vertex AI with gcloud ADC on GCP_PROJECT.
    """

    name = "gemini"
    model_id = "gemini-embedding-2"

    def __init__(self) -> None:
        self._img: dict[str, list[float]] = {}
        self.key = None
        for name in ("GOOGLE_GENERATIVE_AI_API_KEY", "GEMINI_API_KEY"):
            try:
                self.key = env(name)
                break
            except RuntimeError:
                pass
        self.project = os.environ.get("GCP_PROJECT", "")

    def _embed(self, parts: list[dict]) -> list[float]:
        body = {"content": {"parts": parts}}
        throttle("embed")
        if self.key:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:embedContent"
            r = post_json(url, body, {"x-goog-api-key": self.key})
        else:
            token = subprocess.check_output(["gcloud", "auth", "application-default", "print-access-token"], text=True).strip()
            url = (f"https://aiplatform.googleapis.com/v1/projects/{self.project}/locations/global"
                   f"/publishers/google/models/{self.model_id}:embedContent")
            r = post_json(url, body, {"authorization": f"Bearer {token}", "x-goog-user-project": self.project})
        return r["embedding"]["values"]

    def image(self, path: str) -> list[float]:
        if path not in self._img:
            self._img[path] = self._embed([{"inline_data": {"mime_type": "image/jpeg", "data": image_jpeg_b64(path, 1024)}}])
        return self._img[path]

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not self.key:
            return [self._embed([{"text": t}]) for t in texts]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_id}:batchEmbedContents"
        reqs = [{"model": f"models/{self.model_id}", "content": {"parts": [{"text": t}]}} for t in texts]
        throttle("embed")
        return [e["values"] for e in post_json(url, {"requests": reqs}, {"x-goog-api-key": self.key})["embeddings"]]

    def score(self, path: str, texts: list[str]) -> list[dict]:
        img = self.image(path)
        out = []
        for v in self._embed_texts(texts):
            dot = sum(a * b for a, b in zip(img, v))
            norm = (sum(a * a for a in img) * sum(b * b for b in v)) ** 0.5
            cos = round(dot / norm, 4)
            out.append({"cos": cos, "match": cos})
        return out


_scorers: dict[str, object] = {}


def get_scorer(name: str):
    if name not in _scorers:
        _scorers[name] = {"gemini": GeminiScorer, "siglip": SiglipScorer}[name]()
    return _scorers[name]


# ---------------------------------------------------------------- the loop

OPENERS = re.compile(r"^(the|this)\s+(image|photo|picture)\s+(shows|features|depicts|displays)\s+|^(an?\s+)?(image|photo|picture)\s+of\s+", re.I)


def clean(cands: list) -> list[str]:
    seen, out = set(), []
    for c in cands:
        c = re.sub(r"\s+", " ", str(c)).strip().strip('"').strip()
        c = OPENERS.sub("", c)
        c = c[:1].upper() + c[1:]
        if any(b in c.lower() for b in config.OWN_BRANDS):  # never put your own brand in alt text
            continue
        if c and c.lower() not in seen:
            seen.add(c.lower())
            out.append(c)
    return out


def optimize(product: dict, out_dir: Path | None = None, **opts) -> dict:
    o = {**DEFAULTS, **opts}
    scorer = get_scorer(o["scorer"])
    path = product["image_file"]
    t0 = time.time()
    usage = {"input": 0, "output": 0, "calls": 0}

    def add_usage(u: dict) -> None:
        usage["input"] += u["input"]
        usage["output"] += u["output"]
        usage["calls"] += 1

    steps = []
    # Score every distinct alt already on the image's rows; the best one is the baseline ("original").
    currents = [a for a in (product.get("current_alts") or [product["current_alt"]]) if a] or [product["name"]]
    scored = sorted(zip(scorer.score(path, currents), currents), key=lambda x: -x[0]["match"])
    base, base_text = scored[0]
    steps.append({"step": 0, "round": "original", "text": base_text, "chars": len(base_text), **base})
    other_currents = [{"text": t, **sc} for sc, t in scored[1:]]

    # Already at target: still try to improve, but with a fixed budget of writer calls instead of stopping early.
    already_met = base["match"] >= o["target"]

    image = image_jpeg_b64(path, 1024)
    # First vision call: description + first candidates.
    first, u = call_writer(
        o["writer"],
        VISION_PROMPT.format(shop=config.SHOP, name=product["name"], copy=product["copy"][:2500], n=o["candidates"]),
        image,
    )
    add_usage(u)
    description = first.get("description", "")

    def add_round(label: str, cands: list[str]) -> None:
        cands = [c for c in clean(cands) if c.lower() not in {s["text"].lower() for s in steps}]
        for c, s in zip(cands, scorer.score(path, cands) if cands else []):
            steps.append({"step": len(steps), "round": label, "text": c, "chars": len(c), **s})

    add_round("vision", first.get("candidates", []))
    # The current alt competes too: if nothing beats it, the answer is "keep what is there".
    eligible = lambda: [s for s in steps if len(s["text"] or "") <= MAX_ALT_CHARS]
    best = max(eligible(), key=lambda s: s["match"])
    rounds = 0
    while (rounds < o["calls_if_met"] - 1) if already_met else (best["match"] < o["target"] and rounds < o["max_rounds"]):
        rounds += 1
        top = sorted(steps, key=lambda s: -s["match"])[:5]
        history = "\n".join(f"- {s['match']:.3f}  {s['text']}" for s in top)
        nxt, u = call_writer(
            o["writer"],
            REWRITE_PROMPT.format(shop=config.SHOP, description=description, name=product["name"], history=history, n=o["candidates"]),
            image,
        )
        add_usage(u)
        add_round(f"rewrite {rounds}", nxt.get("candidates", []))
        best = max(eligible(), key=lambda s: s["match"])

    price_in, price_out = PRICES.get(o["writer"], (0, 0))
    writer_usd = (usage["input"] * price_in + usage["output"] * price_out) / 1e6
    embed_usd = (EMBED_IMAGE_USD + sum(len(s["text"] or "") for s in steps) * EMBED_TEXT_USD_PER_CHAR) if o["scorer"] == "gemini" else 0
    result = {
        "id": product["id"],
        "options": o,
        "scorer_model": getattr(scorer, "model_id", "multimodalembedding@001"),
        "description": description,
        "steps": steps,
        "before": steps[0],
        "best": best,
        "reached_target": best["match"] >= o["target"],
        "kept_original": best["step"] == 0,
        "already_met": already_met,
        "other_current_alts": other_currents,
        "rewrite_rounds": rounds,
        "usage": usage,
        "cost_usd": round(writer_usd + embed_usd, 6),
        "cost_breakdown": {"writer": round(writer_usd, 6), "embeddings": round(embed_usd, 6)},
        "seconds": round(time.time() - t0, 1),
        "ran_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = out_dir or RESULTS
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{product['id']}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def products() -> list[dict]:
    return json.loads((DATA / "products.json").read_text())


if __name__ == "__main__":
    import sys

    only = {a for a in sys.argv[1:] if a != "all"}
    todo = products()
    if only:
        todo = [p for p in todo if str(p["id"]) in only]
    elif "all" not in sys.argv[1:]:
        todo = todo[:QUICK_RUN]
    for p in todo:
        try:
            r = optimize(p)
            print(f"{p['id']:>8} {p['name'][:30]:30} {to_pct(r['before']['match']):5.1f}% -> {to_pct(r['best']['match']):5.1f}%"
                  f"  rounds={r['rewrite_rounds']}  ${r['cost_usd']:.5f}  {r['seconds']}s")
        except Exception as e:  # keep going; one bad image should not stop the batch
            print(f"{p['id']:>8} FAILED: {e}")
