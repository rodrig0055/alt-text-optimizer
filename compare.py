"""Run several writers over the test subset and compare best match, cost and speed.

  uv run python compare.py [writer ...]

Results land in data/experiments/<writer>/<id>.json (the viewer reads them) and data/experiments/summary.json.
"""

import json
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor

import optimizer as o

WRITERS = sys.argv[1:] or [
    "openai:gpt-5.4-nano",
    "openai:gpt-5.4-mini",
    "gemini:gemini-2.5-flash-lite",
    "gemini:gemini-3.1-flash-lite",
    "gemini:gemini-3.8-flash",
]
EXP = o.DATA / "experiments"


def run_writer(writer: str) -> list[dict]:
    out = EXP / writer.replace(":", "__")
    rows = []
    for p in o.products()[: o.QUICK_RUN]:
        try:
            r = o.optimize(p, out_dir=out, writer=writer)
            rows.append({"id": p["id"], "before": r["before"]["match"], "best": r["best"]["match"],
                         "kept": r["kept_original"], "cost": r["cost_usd"], "seconds": r["seconds"],
                         "text": r["best"]["text"]})
        except Exception as e:  # one failing writer/product should not stop the comparison
            rows.append({"id": p["id"], "error": str(e)[:300]})
    return rows


def main() -> None:
    o.get_scorer("gemini")  # build the shared scorer once, before the threads start
    with ThreadPoolExecutor(len(WRITERS)) as pool:
        results = dict(zip(WRITERS, pool.map(run_writer, WRITERS)))

    summary = {}
    print(f"{'writer':32} {'avg best':>8} {'avg gain':>8} {'$/image':>9} {'s/image':>7}  errors")
    for w, rows in results.items():
        ok = [r for r in rows if "error" not in r]
        if ok:
            summary[w] = {
                "avg_best": round(st.mean(r["best"] for r in ok), 4),
                "avg_gain": round(st.mean(r["best"] - r["before"] for r in ok), 4),
                "avg_cost": round(st.mean(r["cost"] for r in ok), 6),
                "avg_seconds": round(st.mean(r["seconds"] for r in ok), 1),
                "rows": rows,
            }
            s = summary[w]
            print(f"{w:32} {s['avg_best']:8.4f} {s['avg_gain']:+8.4f} {s['avg_cost']:9.5f} {s['avg_seconds']:7.1f}  {len(rows) - len(ok)}")
        else:
            summary[w] = {"rows": rows}
            print(f"{w:32} all failed: {rows[0]['error']}")
    (EXP / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nBest text per product:")
    for pid in [p["id"] for p in o.products()[: o.QUICK_RUN]]:
        scored = [(r["best"], w, r["text"]) for w, rows in results.items() for r in rows if r["id"] == pid and "error" not in r]
        if scored:
            b, w, t = max(scored)
            print(f"  {pid}: {b:.4f} {w}\n      {t}")


if __name__ == "__main__":
    main()
