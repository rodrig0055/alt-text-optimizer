"""Shop settings, from the environment (or a .env file next to this one).

SHOP_DESCRIPTION  how the writer should think of the shop, e.g. "Acme Florals, an online flower shop"
OWN_BRANDS        comma-separated names never allowed in alt text (your own brand names)
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent


def _load_dotenv() -> None:
    f = ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

SHOP = os.environ.get("SHOP_DESCRIPTION", "Acme Florals, an online flower and gift shop")
OWN_BRANDS = tuple(b.strip().lower() for b in os.environ.get("OWN_BRANDS", "acme florals,acmeflorals").split(",") if b.strip())
