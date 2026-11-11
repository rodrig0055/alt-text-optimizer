"""Write the results as a CSV to review or import into your CMS.

  uv run python export.py            # -> data/alt_text.csv

A new alt replaces the current one only when it gains at least MIN_GAIN points, so noise-level
"improvements" never churn a good human-written alt.
"""

import csv
import json

import optimizer as o

MIN_GAIN = 1.0  # points on the 0-100 scale


def main() -> None:
    rows = []
    for p in o.products():
        f = o.RESULTS / f"{p['id']}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        before, best = o.to_pct(r["before"]["match"]), o.to_pct(r["best"]["match"])
        change = not r["kept_original"] and best - before >= MIN_GAIN
        rows.append({
            "id": p["id"], "name": p["name"], "current_alt": p["current_alt"],
            "new_alt": r["best"]["text"] if change else p["current_alt"],
            "changed": "yes" if change else "no",
            "match_before": f"{before:.1f}", "match_after": f"{best if change else before:.1f}",
            "cost_usd": r["cost_usd"],
        })
    out = o.DATA / "alt_text.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows, {sum(r['changed'] == 'yes' for r in rows)} changed -> {out}")


if __name__ == "__main__":
    main()
