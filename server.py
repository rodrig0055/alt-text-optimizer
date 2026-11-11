"""Local viewer for the alt-text optimizer.

  uv run python server.py        # then open http://localhost:8777 (PORT env to change)

GET  /api/state        products + latest results + ratings + defaults
POST /api/run          {"id": 123, "writer": "...", "scorer": "...", "target": 0.8}  re-runs one product
POST /api/rate         {"id": 123, "verdict": "accurate"|"minor"|"wrong", "note": "..."}
"""

import json
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import optimizer

ROOT = Path(__file__).parent
RATINGS = optimizer.DATA / "ratings.json"
lock = threading.Lock()  # one optimization at a time; the scorer models are not thread-safe


def state() -> dict:
    results = {}
    for f in optimizer.RESULTS.glob("*.json"):
        r = json.loads(f.read_text())
        results[r["id"]] = r
    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {}
    # writer comparison runs (compare.py): {writer: {id: result}}
    experiments = {}
    for d in sorted((optimizer.DATA / "experiments").glob("*__*")):
        writer = d.name.replace("__", ":", 1)
        experiments[writer] = {}
        for f in d.glob("*.json"):
            r = json.loads(f.read_text())
            experiments[writer][r["id"]] = {"best": r["best"]["match"], "text": r["best"]["text"], "kept": r["kept_original"],
                                            "cost": r["cost_usd"], "seconds": r["seconds"], "description": r["description"]}
    return {"products": optimizer.products(), "results": results, "ratings": ratings, "experiments": experiments,
            "subset": [p["id"] for p in optimizer.products()[: optimizer.QUICK_RUN]], "scale": {"a": optimizer.SCALE_A, "b": optimizer.SCALE_B},
            "defaults": optimizer.DEFAULTS, "writers": list(optimizer.PRICES), "scorers": ["gemini", "siglip"]}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/state":
            return self.send_json(state())
        if self.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("content-length", 0))) or "{}")
        try:
            if self.path == "/api/run":
                product = next(p for p in optimizer.products() if str(p["id"]) == str(body["id"]))
                opts = {k: body[k] for k in ("writer", "scorer", "target", "max_rounds") if body.get(k) is not None}
                with lock:
                    return self.send_json(optimizer.optimize(product, **opts))
            if self.path == "/api/rate":
                with lock:
                    ratings = json.loads(RATINGS.read_text()) if RATINGS.exists() else {}
                    ratings[str(body["id"])] = {"verdict": body["verdict"], "note": body.get("note", "")}
                    RATINGS.write_text(json.dumps(ratings, indent=2, ensure_ascii=False))
                return self.send_json({"ok": True})
        except Exception as e:
            return self.send_json({"error": str(e)}, 500)
        self.send_json({"error": "not found"}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8777))
    print(f"http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
