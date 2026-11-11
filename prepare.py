"""Turn a catalog CSV into data/products.json and download each product's image.

  uv run python prepare.py examples/demo.csv

CSV columns (header row required):
  id           your product or image id
  name         product name
  image        https URL or a local file path
  current_alt  the alt text live today (may be empty)
  copy         product page copy; used only to name things correctly, never to add what the photo lacks
  url          optional product page link, shown in the viewer
"""

import csv
import json
import shutil
import sys
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
IMAGES = DATA / "images"


def fetch(src: str, dest: Path) -> None:
    if dest.exists():
        return
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(src, headers={"user-agent": "alt-text-optimizer/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        shutil.copyfile(src, dest)


def suffix(src: str) -> str:
    ext = Path(src.split("?")[0]).suffix.lower()
    return ext if ext in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = {"id", "name", "image"} - set(rows[0] if rows else {})
    if missing:
        raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def alt_kind(alt: str, name: str) -> str:
    """empty, placeholder (short or just the product name) or written."""
    if not alt:
        return "empty"
    if len(alt) < 40 or alt.lower().startswith(name.lower().strip()):
        return "placeholder"
    return "written"


def main(path: str) -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    products = []
    for row in read_rows(path):
        pid = row["id"].strip()
        dest = IMAGES / f"{pid}{suffix(row['image'])}"
        fetch(row["image"].strip(), dest)
        products.append({
            "id": pid,
            "name": row["name"].strip(),
            "image_file": f"images/{dest.name}",
            "current_alt": (row.get("current_alt") or "").strip(),
            "alt_kind": alt_kind((row.get("current_alt") or "").strip(), row["name"]),
            "url": (row.get("url") or "").strip(),
            "copy": (row.get("copy") or "").strip(),
        })
        print(f"{pid:>8} {row['name'][:40]}")
    (DATA / "products.json").write_text(json.dumps(products, indent=2, ensure_ascii=False))
    print(f"{len(products)} products -> data/products.json")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/demo.csv")
