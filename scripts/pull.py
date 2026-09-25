"""Freeze a generative training run's split + gold + baseline predictions as a span bake-off fixture.

The run pins the labeling task (typed schema), the held-out documents (``eval_shas``) and the model it
produced; its eval dump lists every held-out unit with any wrong field (``pred`` null on invalid JSON).
A unit absent from the dump was right on every field, so its baseline is the gold.

    python -m scripts.pull 21 --ground <store-url>
"""

from __future__ import annotations

import argparse
import ast
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import httpx

ROOT = Path(__file__).resolve().parents[1]
PAGE_SIZE = 1000
# The baseline's prediction when its generation did not parse — wrong even where gold is null.
INVALID_OUTPUT = "<invalid output>"


def get(client: httpx.Client, path: str, **params) -> dict | list:
    response = client.get(path, params=params)
    response.raise_for_status()
    return response.json()


def all_rows(client: httpx.Client, task_id: int) -> list[dict]:
    rows: list[dict] = []
    while True:
        page = get(client, f"/v1/studio/tasks/{task_id}/rows", limit=PAGE_SIZE, offset=len(rows))
        rows += page["rows"]
        if len(rows) >= page["total"]:
            return rows


def gold(row: dict, fields: list[dict]) -> tuple[dict, list[str], int]:
    """(cells by field name, the fields actually labelled, id/name key conflicts).

    A full label keys cells by field id. A correction row keys ONLY the corrected fields by name — its
    other fields are unlabelled, not null, so they are reported separately rather than read as null."""
    out, labelled, conflicts = {}, [], 0
    full = any(f["id"] in row["cells"] for f in fields)
    for f in fields:
        by_id, by_name = row["cells"].get(f["id"]), row["cells"].get(f["name"])
        if f["id"] in row["cells"] and f["name"] in row["cells"] and by_id != by_name:
            conflicts += 1
        out[f["name"]] = by_id if f["id"] in row["cells"] else by_name
        if full or f["name"] in row["cells"]:
            labelled.append(f["name"])
    return out, labelled, conflicts


def pull(client: httpx.Client, run_id: int, workers: int) -> dict:
    run = get(client, f"/v1/datasets/runs/{run_id}")
    model = next(m for m in get(client, "/v1/ml/models") if m["run_id"] == run_id)
    task = get(client, f"/v1/studio/tasks/{run['labeling_task_id']}")
    dump = get(client, f"/v1/ml/models/{model['model_name']}/{model['model_version']}/eval-dump")
    fields = [{k: f[k] for k in ("id", "name", "type", "desc")} for f in task["schema"]]

    eval_shas = set(ast.literal_eval(run["eval_shas"]) if isinstance(run["eval_shas"], str) else run["eval_shas"])
    rows = [r for r in all_rows(client, task["id"]) if r["status"] != "empty"]

    def source(row: dict) -> str:
        return get(client, f"/v1/studio/tasks/{task['id']}/rows/{quote(row['id'], safe='')}/source")["markdown"] or ""

    with ThreadPoolExecutor(workers) as pool:
        texts = list(pool.map(source, rows))

    wrong = {(item["key"], item["index"]): item for item in dump["items"]}
    units, conflicts = [], 0
    for row, text in zip(rows, texts):
        # Row ids come as ``sha#sN`` and (older) ``sha:N``; ``span`` (``§N``) carries the index either way.
        sha, index = row["id"][:64], row["span"].lstrip("§")
        cells, labelled, n = gold(row, fields)
        conflicts += n
        unit = {"id": row["id"], "collection": row["collection"], "text": text, "gold": cells,
                "labelled": labelled, "split": "eval" if sha in eval_shas else "train"}
        if unit["split"] == "eval":
            # The dump names the wrong fields and carries ``pred`` for those only (null on invalid JSON);
            # every field it does not name was right, i.e. equal to the gold.
            miss = wrong.get((sha, int(index)))
            unit["baseline"] = {
                name: value if miss is None or name not in miss["wrong"]
                else miss["pred"][name] if miss["pred"] else INVALID_OUTPUT
                for name, value in cells.items()
            }
        units.append(unit)

    held_out = {u["collection"] for u in units if u["split"] == "eval"}
    leaked = held_out & {u["collection"] for u in units if u["split"] == "train"}
    assert not leaked, f"collections on both sides of the split: {leaked}"
    return {
        "run_id": run_id,
        "task": {"id": task["id"], "name": task["name"], "unit": task["unit"], "fields": fields},
        "baseline": {"model": f"{model['model_name']}/{model['model_version']}", "backend": model["backend"],
                     "status": model["status"], "n_eval_dump": dump["metrics"]["n_eval"]},
        "holdout": run["holdout"],
        "gold_key_conflicts": conflicts,
        "units": units,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_ids", type=int, nargs="+", help="training run ids (ground /v1/datasets/runs)")
    parser.add_argument("--ground", required=True, help="base URL of the store to pull from")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    out_dir = ROOT / "fixtures"
    out_dir.mkdir(exist_ok=True)
    with httpx.Client(base_url=args.ground, timeout=60) as client:
        for run_id in args.run_ids:
            fixture = pull(client, run_id, args.workers)
            path = out_dir / f"run-{run_id}.json"
            path.write_text(json.dumps(fixture, ensure_ascii=False, indent=1))
            n_eval = sum(u["split"] == "eval" for u in fixture["units"])
            print(f"run {run_id}: {fixture['task']['name']!r} vs {fixture['baseline']['model']} — "
                  f"{len(fixture['units']) - n_eval} train / {n_eval} eval units "
                  f"(dump n_eval={fixture['baseline']['n_eval_dump']}), "
                  f"{fixture['gold_key_conflicts']} gold key conflicts, "
                  f"{sum(len(u['labelled']) < len(fixture['task']['fields']) for u in fixture['units'])} "
                  f"partially labelled → {path.name}")


if __name__ == "__main__":
    main()
