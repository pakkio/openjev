"""Catalogue today's BBC News stories into openjev's 8 news sections, with Jev as a comparison.

  fetch                 pull the BBC News + Sport RSS feeds into data/bbc/DATE.jsonl (one row per story,
                        de-duplicated by link). Stories from the Business, Politics, Health, Science &
                        Environment, Technology, Entertainment & Arts and Sport feeds carry those sections
                        as reference labels; World, UK, Education and top stories are catalogued unlabelled.
  classify openjev|jev  score every story with openjev (Gemma in the chat format that lists the options, as
                        LoRA training and serving use it; --adapter adds a LoRA adapter) or with TypeSafe's
                        Jev, into runs/bbc/DATE-ENGINE.jsonl (ENGINE: jev, openjev or openjev-ADAPTERNAME)
  report                runs/bbc/DATE.md: the catalogue grouped by openjev's section, with Jev's pick and
                        accuracy of both against the BBC's own sections

usage: bbc_catalogue.py fetch|report [--date D]
       bbc_catalogue.py classify openjev [--adapter DIR] [--model M] [--quantize 4bit] [--date D]
       bbc_catalogue.py classify jev [--date D]
"""
import argparse, datetime, json, sys, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

CATEGORIES = ["politics", "sports", "business", "technology", "health", "science", "entertainment", "crime"]
FEEDS = {  # feed path -> openjev section it stands for (None: catalogue only)
    "news/rss.xml": None, "news/world/rss.xml": None, "news/uk/rss.xml": None, "news/education/rss.xml": None,
    "news/business/rss.xml": "business", "news/politics/rss.xml": "politics", "news/health/rss.xml": "health",
    "news/science_and_environment/rss.xml": "science", "news/technology/rss.xml": "technology",
    "news/entertainment_and_arts/rss.xml": "entertainment", "sport/rss.xml": "sports",
}


def context(title: str, description: str) -> str:
    return f"Classify this news story into a newspaper section.\nHeadline: {title}\nSummary: {description}"


def fetch(path: Path) -> None:
    stories: dict[str, dict] = {}
    for feed, section in FEEDS.items():
        req = urllib.request.Request("https://feeds.bbci.co.uk/" + feed, headers={"User-Agent": "openjev/0.1"})
        for item in ET.fromstring(urllib.request.urlopen(req, timeout=30).read()).findall("./channel/item"):
            link = (item.findtext("link") or "").split("?")[0]
            s = stories.setdefault(link, {"title": (item.findtext("title") or "").strip(),
                                          "description": (item.findtext("description") or "").strip(),
                                          "link": link, "published": item.findtext("pubDate"),
                                          "feeds": [], "sections": []})
            s["feeds"].append(feed)
            if section and section not in s["sections"]:
                s["sections"].append(section)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in stories.values():
            labels = [CATEGORIES.index(c) for c in s["sections"]]
            f.write(json.dumps({"context": context(s["title"], s["description"]), "options": CATEGORIES,
                                "label": labels[0] if labels else -1, "labels": labels, **s},
                               ensure_ascii=False) + "\n")
    n_lab = sum(1 for s in stories.values() if s["sections"])
    print(f"{len(stories)} stories ({n_lab} with a BBC section) -> {path}")


def classify(rows: list[dict], engine: str, args) -> list[list[float]]:
    if engine == "jev":
        from jev_eval import api_key, ask
        from concurrent.futures import ThreadPoolExecutor
        key = api_key()
        with ThreadPoolExecutor(8) as pool:
            return [r["probabilities"] for r in pool.map(lambda r: ask(r, key, "jev-latest"), rows)]
    # The LoRA engine's chat format puts the option list in the user turn; OptionScorer's chat format
    # scores the section names as free replies without showing them, which is far weaker here.
    from openjev.lora_serve import LoraEngine
    adapters = {"a": args.adapter} if args.adapter else {}
    engine_ = LoraEngine(args.model, adapters, args.quantize)
    out = []
    for i, r in enumerate(rows, 1):
        out.append(engine_.score(r["context"], r["options"], adapter="a" if adapters else None)[1])
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", file=sys.stderr, flush=True)
    return out


def report(date: str) -> None:
    rows = [json.loads(l) for l in open(f"data/bbc/{date}.jsonl")]
    preds = {}
    for p in sorted(Path("runs/bbc").glob(f"{date}-*.jsonl")):
        preds[p.stem[len(date) + 1:]] = [json.loads(l)["probabilities"] for l in open(p)]
    if not preds:
        sys.exit("nothing classified yet")
    main = "openjev" if "openjev" in preds else "jev"
    pick = {e: [max(range(len(p)), key=p.__getitem__) for p in ps] for e, ps in preds.items()}
    lab = [i for i, r in enumerate(rows) if r["labels"]]
    lines = [f"# BBC News catalogue, {date}", "",
             f"{len(rows)} stories from the BBC News and Sport RSS feeds, sorted into openjev's 8 sections "
             f"by {main}. {len(lab)} stories also appear in a BBC section feed, used as the reference below.", "",
             "| engine | agrees with a BBC section | stories |", "|---|---|---|"]
    for e in preds:
        hits = sum(pick[e][i] in rows[i]["labels"] for i in lab)
        lines.append(f"| {e} | {hits / len(lab):.3f} | {len(lab)} |")
    if main != "jev" and "jev" in pick:
        agree = sum(a == b for a, b in zip(pick[main], pick["jev"])) / len(rows)
        lines += ["", f"{main} and Jev pick the same section for {agree:.1%} of all {len(rows)} stories."]
    for c, cat in enumerate(CATEGORIES):
        idx = [i for i in range(len(rows)) if pick[main][i] == c]
        if not idx:
            continue
        lines += ["", f"## {cat} ({len(idx)})", ""]
        for i in sorted(idx, key=lambda i: -preds[main][i][c]):
            r = rows[i]
            notes = [f"p={preds[main][i][c]:.2f}"]
            if r["labels"] and c not in r["labels"]:
                notes.append("BBC: " + "/".join(r["sections"]))
            other = "jev" if main != "jev" else None
            if other in pick and pick[other][i] != c:
                notes.append(f"Jev: {CATEGORIES[pick[other][i]]}")
            lines.append(f"- [{r['title']}]({r['link']}) ({', '.join(notes)})")
    out = Path(f"runs/bbc/{date}.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:9 + len(preds)]))
    print(f"-> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "classify", "report"])
    ap.add_argument("engine", nargs="?", choices=["openjev", "jev"])
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--quantize", default="4bit")
    ap.add_argument("--adapter", help="LoRA adapter dir for classify openjev")
    args = ap.parse_args()
    data = Path(f"data/bbc/{args.date}.jsonl")
    if args.cmd == "fetch":
        fetch(data)
    elif args.cmd == "report":
        report(args.date)
    else:
        if not args.engine:
            sys.exit("classify needs an engine: openjev or jev")
        rows = [json.loads(l) for l in open(data)]
        probs = classify(rows, args.engine, args)
        name = args.engine + (f"-{Path(args.adapter).name.removeprefix('lora-')}" if args.adapter else "")
        out = Path(f"runs/bbc/{args.date}-{name}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for p in probs:
                f.write(json.dumps({"probabilities": p}) + "\n")
        print(f"{len(rows)} stories classified by {name} -> {out}")
