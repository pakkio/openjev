"""Rate 100 IMDb films with a System One score + noul request, and check both engines against IMDb's ratings.

The same request goes to TypeSafe's Jev and to an openjev server (`openjev serve ... --lora ...`, whose
/v1/systemone answers in the option-listing chat format): the film's title, year, genres and runtime as
`state`, a 5-level `score` question on how highly IMDb users rate it (levels = IMDb rating bands), and a
`noul` on whether it is rated 7.0 or higher. data/imdb/movies100.jsonl holds the films and their live
IMDb ratings (IMDb's daily non-commercial datasets); responses are cached in runs/imdb/ENGINE.jsonl.

usage: imdb_rate.py jev | openjev [--url http://127.0.0.1:8012] | report
"""
import json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BANDS = [(0, 5.5), (5.5, 6.5), (6.5, 7.2), (7.2, 8.0), (8.0, 10.1)]
LEVELS = ["Poor: IMDb rating below 5.5", "Mediocre: IMDb rating 5.5 to 6.4", "Decent: IMDb rating 6.5 to 7.1",
          "Good: IMDb rating 7.2 to 7.9", "Excellent: IMDb rating 8.0 or higher"]
QUESTIONS = {
    "rating": {"type": "score", "instructions": "How highly do IMDb users rate this film?", "criteria": LEVELS},
    "good": {"type": "noul", "instructions": "Do IMDb users rate this film 7.0 or higher?"},
}


def band(rating: float) -> int:
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo <= rating < hi)


def body(film: dict) -> dict:
    state = {k: film[k] for k in ("title", "year", "genres", "runtime_minutes")}
    return {"state": state, "model": "jev-latest", "questions": QUESTIONS}


def post(url: str, payload: dict, headers: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"content-type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def run(engine: str, url: str | None) -> None:
    films = [json.loads(l) for l in open("data/imdb/movies100.jsonl")]
    if engine == "jev":
        from jev_eval import URL, api_key
        target, headers = URL, {"Authorization": f"Bearer {api_key()}"}
    else:
        target, headers = (url or "http://127.0.0.1:8012") + "/v1/systemone", {}
    with ThreadPoolExecutor(8 if engine == "jev" else 1) as pool:
        answers = list(pool.map(lambda f: post(target, body(f), headers)["answers"], films))
    out = Path(f"runs/imdb/{engine}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for f, a in zip(films, answers):
            fh.write(json.dumps({"tconst": f["tconst"], "score": a["rating"]["score"],
                                 "probabilities": a["rating"]["probabilities"], "noul": a["good"]["noul"]}) + "\n")
    print(f"{len(films)} films rated by {engine} -> {out}")


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=x.__getitem__)
        r = [0.0] * len(x)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return cov / (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5


def auc(scores: list[float], truth: list[bool]) -> float:
    pos = [s for s, t in zip(scores, truth) if t]
    neg = [s for s, t in zip(scores, truth) if not t]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def report() -> None:
    films = [json.loads(l) for l in open("data/imdb/movies100.jsonl")]
    truth_band = [band(f["rating"]) for f in films]
    truth_good = [f["rating"] >= 7.0 for f in films]
    print(f"{len(films)} films, IMDb ratings {min(f['rating'] for f in films)}-{max(f['rating'] for f in films)}, "
          f"{sum(truth_good)} rated 7.0+")
    print("| engine | score: Spearman vs IMDb rating | score: exact band | score: mean abs error (bands) "
          "| noul: accuracy | noul: AUC | noul: Brier |")
    print("|---|---|---|---|---|---|---|")
    for path in sorted(Path("runs/imdb").glob("*.jsonl")):
        res = [json.loads(l) for l in open(path)]
        s = [r["score"] for r in res]
        exact = sum(max(r["probabilities"], key=lambda k: r["probabilities"][k]) == str(b)
                    for r, b in zip(res, truth_band)) / len(res)
        mae = sum(abs(x - b) for x, b in zip(s, truth_band)) / len(res)
        p = [r["noul"] for r in res]
        acc = sum((x >= 0.5) == t for x, t in zip(p, truth_good)) / len(res)
        brier = sum((x - t) ** 2 for x, t in zip(p, truth_good)) / len(res)
        print(f"| {path.stem} | {spearman(s, [f['rating'] for f in films]):.3f} | {exact:.2f} | {mae:.2f} "
              f"| {acc:.2f} | {auc(p, truth_good):.3f} | {brier:.3f} |")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "report":
        report()
    else:
        run(cmd, sys.argv[3] if len(sys.argv) > 3 and sys.argv[2] == "--url" else None)
