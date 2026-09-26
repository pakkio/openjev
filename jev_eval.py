"""Score an openjev JSONL set with TypeSafe's hosted Jev, for an openjev-vs-Jev comparison.

Each row {context, options, label} becomes one System One request: the context is the `state` and
the options are the criteria of a single `choice` question, so Jev and openjev see the same text
and pick among the same options. Reports top-1, top-3 and 10-bin ECE on Jev's choice
probabilities, the same metrics as `openjev lora-eval`.

Responses are cached in runs/jev/<name>.jsonl (one line per row index), so a rerun only sends the
rows it has not scored yet. The key comes from TYPESAFE_API_KEY, else from ../.env.

usage: jev_eval.py DATA.jsonl [--limit N] [--name NAME] [--workers 8] [--model jev-latest]
"""
import argparse, json, os, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://api.typesafe.ai/v1/systemone"
INSTRUCTIONS = "Choose the option that correctly answers the request in the state."


def api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key and Path("../.env").exists():
        for line in open("../.env"):
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
    if not key:
        sys.exit("set TYPESAFE_API_KEY (or put it in ../.env)")
    return key


def request(body: dict, key: str, tries: int = 6) -> dict:
    data = json.dumps(body).encode()
    for attempt in range(tries):
        req = urllib.request.Request(URL, data=data, headers={"Authorization": f"Bearer {key}",
                                                               "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 529) or attempt == tries - 1:
                raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
            time.sleep(float(e.headers.get("retry-after") or 2 ** attempt))
        except (urllib.error.URLError, TimeoutError):
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def ask(row: dict, key: str, model: str) -> dict:
    """One choice question; returns {probabilities: [p per option, in row order], usage, seconds}."""
    keyed = len(set(row["options"])) == len(row["options"])
    names = row["options"] if keyed else [f"option {i + 1}" for i in range(len(row["options"]))]
    criteria = {n: (None if keyed else o) for n, o in zip(names, row["options"])}
    body = {"state": row["context"], "model": model,
            "questions": {"q": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}}}
    t = time.perf_counter()
    out = request(body, key)
    probs = out["answers"]["q"]["probabilities"]
    return {"probabilities": [float(probs.get(n, 0.0)) for n in names], "model": out.get("model"),
            "usage": out.get("usage", {}), "seconds": round(time.perf_counter() - t, 3)}


def metrics(rows: list[dict], preds: list[dict]) -> dict:
    top1 = top3 = 0
    bins = [[0, 0.0, 0.0] for _ in range(10)]
    for row, p in zip(rows, preds):
        probs = p["probabilities"]
        order = sorted(range(len(probs)), key=lambda i: -probs[i])
        hit = order[0] == row["label"]
        top1 += hit
        top3 += row["label"] in order[:3]
        pm = max(probs)
        b = bins[min(9, int(pm * 10))]
        b[0] += 1; b[1] += pm; b[2] += hit
    n = len(rows)
    ece = sum(abs(sc - sh) for c, sc, sh in bins if c) / n
    return {"top1": top1 / n, "top3": top3 / n, "ece": ece, "examples": n,
            "input_tokens": sum(p["usage"].get("input_tokens", 0) for p in preds),
            "median_seconds": sorted(p["seconds"] for p in preds)[n // 2]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", help="cache/report name (default: the data file's parent dir)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="jev-latest")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    name = args.name or Path(args.data).parent.name
    cache_path = Path("runs/jev") / f"{name}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_path.exists():
        for line in open(cache_path):
            r = json.loads(line)
            cache[r["i"]] = r
    todo = [i for i in range(len(rows)) if i not in cache]
    key = api_key()
    print(f"{name}: {len(rows)} rows, {len(rows) - len(todo)} cached, sending {len(todo)}", file=sys.stderr)
    with open(cache_path, "a") as f, ThreadPoolExecutor(args.workers) as pool:
        for i, res in zip(todo, pool.map(lambda i: ask(rows[i], key, args.model), todo)):
            cache[i] = {"i": i, **res}
            f.write(json.dumps(cache[i]) + "\n")
    report = {"name": name, "data": args.data, "model": cache[0].get("model") if cache else None,
              **metrics(rows, [cache[i] for i in range(len(rows))])}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
