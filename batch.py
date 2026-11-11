"""Full catalog run: every product in data/products.json, resumable, paced for Google's free tier.

  nohup uv run python batch.py > data/run.log 2>&1 &

Results: data/results/<id>.json (existing ones are skipped, so re-running resumes).
Errors:  data/errors.jsonl. If the daily quota runs out the run stops; start it again tomorrow.
"""

import json
import time

import optimizer as o

ROOT = o.DATA
RESULTS = o.RESULTS


def main() -> None:
    images = o.products()
    RESULTS.mkdir(parents=True, exist_ok=True)
    todo = [i for i in images if not (RESULTS / f"{i['id']}.json").exists()]
    print(f"{len(images)} images, {len(images) - len(todo)} done, {len(todo)} to go "
          f"(writer {o.DEFAULTS['writer']}, {o.RPM['writer']:.0f} writer rpm)", flush=True)

    t0, done, spent = time.time(), 0, 0.0
    for img in todo:
        try:
            r = o.optimize(img, out_dir=RESULTS)
        except o.DailyQuotaExhausted as e:
            print(f"DAILY QUOTA EXHAUSTED after {done} images this run; resume tomorrow.\n{e}", flush=True)
            break
        except Exception as e:  # keep going; the export lists anything that never finished
            with open(ROOT / "errors.jsonl", "a") as f:
                f.write(json.dumps({"id": img["id"], "error": str(e)[:500], "at": time.strftime("%F %T")}) + "\n")
            print(f"{img['id']:>6} ERROR {str(e)[:120]}", flush=True)
            continue
        done += 1
        spent += r["cost_usd"]
        rate = done / max(1, time.time() - t0) * 3600
        print(f"{img['id']:>6} {o.to_pct(r['before']['match']):5.1f}% -> {o.to_pct(r['best']['match']):5.1f}%"
              f"  calls={r['usage']['calls']}  {'keep' if r['kept_original'] else 'new '}"
              f"  [{done}/{len(todo)}, {rate:.0f}/h, ${spent:.3f} at paid rates]", flush=True)
    print(f"finished this run: {done} images in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
