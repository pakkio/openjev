"""HTTP server: one loaded model, many scoring requests.

    openjev serve --host 0.0.0.0 --port 8000
    curl -s localhost:8000/score -H 'content-type: application/json' \
      -d '{"context": "The capital of France is", "options": [" Paris", " Berlin"]}'
"""
from __future__ import annotations

import os
import time
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .systemone import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    confidence,
    render_choice,
    render_noul,
    render_score,
    system_one,
)


def _resolve_backend(backend: str | None = None) -> str:
    """Resolve the scoring backend.

    Precedence: explicit arg > OPENJEV_BACKEND env var > "mlx".
    """
    if backend and backend in ("mlx", "torch"):
        return backend
    env_backend = os.environ.get("OPENJEV_BACKEND")
    if env_backend in ("mlx", "torch"):
        return env_backend
    return "mlx"


def _get_scorer_class(backend: str):
    """Return the OptionScorer class for the given backend.

    Imported lazily: mlx is Apple-silicon only, so importing it on a CUDA or
    CPU host fails at load time even when the torch backend is the one wanted.
    """
    if backend == "torch":
        from .scorer_torch import OptionScorer as TorchOptionScorer
        return TorchOptionScorer
    from .scorer import OptionScorer as MLXOptionScorer
    return MLXOptionScorer


def _default_model(backend: str) -> str:
    """Each backend ships its own default checkpoint format."""
    if backend == "torch":
        from .scorer_torch import DEFAULT_MODEL
    else:
        from .scorer import DEFAULT_MODEL
    return DEFAULT_MODEL


class ScoreRequest(BaseModel):
    context: str
    options: list[str] = Field(min_length=2)
    norm: Literal["mean", "sum", "pmi"] = "mean"
    chat: bool = False
    sep: str = ""


class OptionOut(BaseModel):
    option: str
    n_tokens: int
    logprob_sum: float
    logprob_mean: float
    logprob_uncond: float | None
    score: float
    probability: float


class ScoreResponse(BaseModel):
    best: str
    best_index: int
    options: list[OptionOut]
    timing: dict[str, float]


def _resolve_quantize(quantize: str | None = None) -> str:
    """Resolve the weight quantisation: explicit arg > OPENJEV_QUANTIZE > none."""
    for candidate in (quantize, os.environ.get("OPENJEV_QUANTIZE")):
        if candidate in ("none", "8bit", "4bit"):
            return candidate
    return "none"


def _resolve_heads(heads: dict | None) -> dict:
    """Checkpoint paths per question type: explicit dict > OPENJEV_HEAD_<TYPE> env > none."""
    out: dict = {}
    for qtype in ("choice", "score", "noul"):
        path = (heads or {}).get(qtype) or os.environ.get(f"OPENJEV_HEAD_{qtype.upper()}")
        if path:
            out[qtype] = path
    return out


def create_app(model_path: str | None = None, batch_size: int = 8, backend: str | None = None,
               quantize: str | None = None, heads: dict | None = None) -> FastAPI:
    backend = _resolve_backend(backend)
    quantize = _resolve_quantize(quantize)
    head_paths = _resolve_heads(heads)
    model_path = model_path or os.environ.get("OPENJEV_MODEL") or _default_model(backend)
    OptionScorer = _get_scorer_class(backend)
    app = FastAPI(title="openjev", version="0.1.0")
    state: dict = {}
    api_key = os.environ.get("OPENJEV_API_KEY")  # if set, /v1/systemone requires "Authorization: Bearer ***"
    model_name = os.path.basename(model_path.rstrip("/"))

    def _auth(authorization: str | None = Header(default=None)) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(401, "invalid or missing API key")

    @app.on_event("startup")
    def _load() -> None:
        t = time.perf_counter()
        kwargs = {"quantize": quantize} if backend == "torch" else {}
        scorer = OptionScorer(model_path, batch_size=batch_size, **kwargs)
        scorer.score("warm up", ["a", "b"])  # compile kernels / warm up before the first request
        state["scorer"] = scorer
        state["heads"] = {}
        if head_paths:
            from .head_serve import HeadScorer

            for qtype, path in head_paths.items():
                state["heads"][qtype] = HeadScorer(scorer, backend, path)
        state["load_s"] = time.perf_counter() - t

    @app.get("/health")
    def health() -> dict:
        return {"ok": "scorer" in state, "model": model_path, "backend": backend,
                "quantize": quantize, "heads": sorted(state.get("heads", {})),
                "load_s": state.get("load_s")}

    def _head_probs(qtype: str, context: str, options: list[str]) -> list[float] | None:
        head = state.get("heads", {}).get(qtype)
        if head is None:
            return None
        try:
            return head.answer(context, options)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        probs = _head_probs("choice", req.context, req.options)
        if probs is not None:  # head mode: softmax over head logits; score=logit-equivalent log-prob
            import math

            best = max(range(len(probs)), key=lambda i: probs[i])
            return ScoreResponse(
                best=req.options[best],
                best_index=best,
                options=[OptionOut(option=o, n_tokens=0, logprob_sum=math.log(max(p, 1e-12)),
                                   logprob_mean=math.log(max(p, 1e-12)), logprob_uncond=None,
                                   score=math.log(max(p, 1e-12)), probability=p)
                         for o, p in zip(req.options, probs)],
                timing={"head_mode": 1.0, **{k: float(v) for k, v in scorer.last_timing.items()}},
            )
        try:
            res = scorer.score(req.context, req.options, norm=req.norm, chat=req.chat, sep=req.sep)
        except ValueError as e:
            raise HTTPException(400, str(e))
        best = max(range(len(res)), key=lambda i: res[i].score)
        return ScoreResponse(
            best=res[best].option,
            best_index=best,
            options=[OptionOut(**r.to_dict()) for r in res],
            timing=scorer.last_timing,
        )

    @app.post("/v1/systemone", response_model=SystemOneResponse, dependencies=[Depends(_auth)])
    def systemone(req: SystemOneRequest) -> SystemOneResponse:
        """TypeSafe System One contract: state + typed questions -> typed answers (docs.typesafe.ai).

        Per-question head mode (Route A): when a checkpoint is configured for the
        question type, the prompt is rendered with the same systemone renderers
        used at train time and answered from the head softmax instead of zero-shot
        next-token likelihoods. NOTE: heads must be trained on these rendered
        prompts; a head trained on a different rendering serves answers that are
        well-formed but off-distribution.
        """
        scorer: OptionScorer = state.get("scorer")
        if scorer is None:
            raise HTTPException(503, "model still loading")
        if not state.get("heads"):
            try:
                return system_one(scorer, req, model_name=req.model or model_name)
            except ValueError as e:
                raise HTTPException(400, str(e))
        from .systemone import ChoiceQuestion, NoulQuestion, ScoreQuestion

        answers: dict = {}
        in_tok = out_tok = 0
        for qid, q in req.questions.items():
            if isinstance(q, ChoiceQuestion):
                prompt, labels = render_choice(req.state, q)
                qtype = "choice"
            elif isinstance(q, ScoreQuestion):
                prompt, labels = render_score(req.state, q)
                qtype = "score"
            elif isinstance(q, NoulQuestion):
                prompt, labels = render_noul(req.state, q)
                qtype = "noul"
            else:
                raise HTTPException(400, f"unknown question type for {qid!r}")
            probs = _head_probs(qtype, prompt, labels)
            if probs is None:  # no head for this type: fall back to zero-shot
                try:
                    res = scorer.score(prompt, labels, norm="sum", chat=False, sep="")
                except ValueError as e:
                    raise HTTPException(400, str(e))
                probs = [r.probability for r in res]
                in_tok += int(scorer.last_timing["context_tokens"])
                out_tok += int(scorer.last_timing["option_tokens"])
            if isinstance(q, ChoiceQuestion):
                best = max(range(len(labels)), key=lambda i: probs[i])
                answers[qid] = ChoiceAnswer(choice=labels[best], probabilities=dict(zip(labels, probs)),
                                            confidence=confidence(probs))
            elif isinstance(q, ScoreQuestion):
                answers[qid] = ScoreAnswer(score=sum(i * p for i, p in enumerate(probs)),
                                           confidence=confidence(probs),
                                           legend={str(i): c for i, c in enumerate(q.criteria)},
                                           probabilities={str(i): p for i, p in enumerate(probs)})
            else:
                answers[qid] = NoulAnswer(noul=probs[0])
        usage = {"input_tokens": in_tok, "output_tokens": out_tok}
        return SystemOneResponse(model=req.model or model_name, answers=answers,
                                 usage=usage)

    return app


def serve(host: str, port: int, model_path: str | None, batch_size: int, backend: str | None = None,
          quantize: str | None = None, heads: dict | None = None) -> None:
    import uvicorn

    uvicorn.run(create_app(model_path, batch_size, backend, quantize, heads), host=host, port=port, workers=1)
