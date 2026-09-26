"""Jev vs openjev (zero-shot and with LoRA adapters) on the same rows of one JSONL set.

Every engine answers the same {context, options, label} rows and gets the same metrics: top-1,
top-3, 10-bin ECE, median seconds per row, and how often it picks the same option as Jev.

  jev                  TypeSafe's hosted Jev, one System One `choice` question per row (jev_eval.py;
                       responses are cached in runs/jev/, so a rerun costs nothing)
  openjev              Gemma zero-shot in the chat format that lists the options, as LoRA training and
                       `openjev serve --lora` use it
  NAME=DIR[:chat|:plain]
                       openjev with that LoRA adapter (float32 or int8). The prompt format must be the one
                       the adapter was trained in: taken from DIR/openjev_lora.json when it records it,
                       else from the suffix, else chat if "chat" is in the directory name, else plain.

One 4-bit base model is loaded for all adapters (lora_serve.LoraEngine). The table goes to stdout and
the numbers to runs/compare/NAME.json.

usage: compare.py DATA.jsonl [--limit N] [--no-jev] [--no-zero-shot] [NAME=DIR[:chat|:plain] ...]
e.g.   compare.py data/synthetic/movies_general/test.jsonl movies=runs/lora-movies_general_v2:plain
"""
import argparse, json, statistics, time
from pathlib import Path

from jev_eval import metrics as _metrics


def adapter_spec(spec: str) -> tuple[str, str, bool]:
    name, _, rest = spec.partition("=")
    path, _, fmt = rest.partition(":")
    if not (name and path):
        raise SystemExit(f"adapter spec must be NAME=DIR[:chat|:plain], got {spec!r}")
    if fmt:
        chat = fmt == "chat"
    elif (Path(path) / "openjev_lora.json").exists() and json.load(open(Path(path) / "openjev_lora.json")).get("chat") is not None:
        chat = bool(json.load(open(Path(path) / "openjev_lora.json"))["chat"])
    else:
        chat = "chat" in Path(path).name
    return name, path, chat


def top(p: list[float]) -> int:
    return max(range(len(p)), key=p.__getitem__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("adapters", nargs="*", help="NAME=DIR[:chat|:plain]")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--no-zero-shot", action="store_true")
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--quantize", default="4bit")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    set_name = Path(args.data).parent.name
    results: dict[str, dict] = {}
    preds: dict[str, list[list[float]]] = {}

    if not args.no_jev:
        import subprocess, sys
        cmd = [sys.executable, "jev_eval.py", args.data, "--name", set_name] + (["--limit", str(args.limit)] if args.limit else [])
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        cache = {r["i"]: r for r in map(json.loads, open(f"runs/jev/{set_name}.jsonl"))}
        preds["jev"] = [cache[i]["probabilities"] for i in range(len(rows))]
        results["jev"] = {**_metrics(rows, [cache[i] for i in range(len(rows))]), "format": "System One choice"}

    specs = [adapter_spec(s) for s in args.adapters]
    if specs or not args.no_zero_shot:
        from openjev.lora_serve import LoraEngine
        engine = LoraEngine(args.model, {n: p for n, p, _ in specs}, args.quantize)
        runs = ([] if args.no_zero_shot else [("openjev", None, True)]) + \
               ([("openjev (plain)", None, False)] if any(not c for _, _, c in specs) and not args.no_zero_shot else []) + \
               [(f"openjev + {n}", n, c) for n, _, c in specs]
        for label, adapter, chat in runs:
            ps, secs = [], []
            for r in rows:
                t = time.perf_counter()
                ps.append(engine.score(r["context"], r["options"], adapter=adapter, chat=chat)[1])
                secs.append(time.perf_counter() - t)
            preds[label] = ps
            m = _metrics(rows, [{"probabilities": p, "usage": {}, "seconds": s} for p, s in zip(ps, secs)])
            m.pop("input_tokens", None)
            results[label] = {**m, "format": "chat" if chat else "plain"}

    if "jev" in preds:
        for label in results:
            results[label]["same_as_jev"] = statistics.fmean(top(a) == top(b) for a, b in zip(preds[label], preds["jev"]))
    print(f"{set_name}: {len(rows)} rows")
    print("| engine | format | top-1 | top-3 | ECE | same pick as Jev | median s/row |")
    print("|---|---|---|---|---|---|---|")
    for label, m in results.items():
        same = f"{m['same_as_jev']:.0%}" if "same_as_jev" in m else "-"
        print(f"| {label} | {m['format']} | {m['top1']:.3f} | {m['top3']:.3f} | {m['ece']:.3f} | {same} | {m['median_seconds']:.2f} |")
    out = Path("runs/compare") / f"{set_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"data": args.data, "rows": len(rows), "results": results}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
