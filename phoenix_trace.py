"""
phoenix_trace.py — Instrumentation de PERFORMANCE du pipeline RAG vers Phoenix
==============================================================================
Complément de phoenix_export.py (qui envoie la QUALITÉ). Ici on remplit la
colonne « Software Performance » de la taxonomie : à chaque appel du pipeline,
on envoie à Phoenix une trace avec :
  - la LATENCE   : durée du span (automatique, gratuite) ;
  - les TOKENS   : prompt / completion / total, SI le pipeline les expose ;
  - input/output : question et réponse ;
  - les contextes récupérés.

Principe : on « enrobe » (wrap) la fonction pipeline sans la modifier. Le tracer
est créé une seule fois et réutilisé pour tous les appels (export par lots).

Dépendances : `pip install -r requirements-phoenix.txt` (SDK OpenTelemetry).
Serveur Phoenix lancé (docker-compose.yml). Compatible Python 3.13.

Usage :
    from rag_evaluation_agent import RagEval
    from phoenix_trace import traced_pipeline

    pipe = traced_pipeline(mon_pipeline, project_name="rag-locatif")
    agent = RagEval(pipeline=pipe, judge_model="gpt-4o-mini")
    # chaque appel agent.pipeline("...") est désormais tracé (latence + tokens)

Extraction des tokens : par défaut on cherche des formes courantes dans le dict
renvoyé par le pipeline (clé "usage" façon OpenAI, ou "token_count"). Si ton
pipeline expose les tokens autrement, passe ton propre `token_extractor`.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("RagEval.phoenix")

OTLP_ENDPOINT_DEFAULT = "http://localhost:4317"

# Conventions OpenInference (en clair, pas de dépendance dure).
ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_INPUT = "input.value"
ATTR_OUTPUT = "output.value"
ATTR_PROJECT = "openinference.project.name"
ATTR_TOK_PROMPT = "llm.token_count.prompt"
ATTR_TOK_COMPLETION = "llm.token_count.completion"
ATTR_TOK_TOTAL = "llm.token_count.total"

TokenTriple = Tuple[Optional[int], Optional[int], Optional[int]]


def _default_token_extractor(result: Dict[str, Any]) -> TokenTriple:
    """Cherche prompt/completion/total dans des formes courantes. Renvoie (p, c, t)."""
    if not isinstance(result, dict):
        return (None, None, None)

    # Forme OpenAI : result["usage"] = {prompt_tokens, completion_tokens, total_tokens}
    usage = result.get("usage") or result.get("token_usage") or {}
    if isinstance(usage, dict) and usage:
        p = usage.get("prompt_tokens") or usage.get("input_tokens")
        c = usage.get("completion_tokens") or usage.get("output_tokens")
        t = usage.get("total_tokens") or (
            (p or 0) + (c or 0) if (p is not None or c is not None) else None
        )
        return (p, c, t)

    # Forme simple : result["token_count"] = int (total) ou dict
    tc = result.get("token_count") or result.get("tokens")
    if isinstance(tc, dict):
        return (tc.get("prompt"), tc.get("completion"), tc.get("total"))
    if isinstance(tc, (int, float)):
        return (None, None, int(tc))

    return (None, None, None)


def _build_tracer(project_name: str, endpoint: str):
    """TracerProvider avec BatchSpanProcessor (adapté à un pipeline appelé souvent)."""
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    resource = Resource.create(
        {"service.name": project_name, ATTR_PROJECT: project_name}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    return provider, provider.get_tracer("rageval-perf")


def traced_pipeline(
    pipeline: Callable[[str], Dict[str, Any]],
    project_name: str = "rageval-perf",
    endpoint: str = OTLP_ENDPOINT_DEFAULT,
    token_extractor: Optional[Callable[[Dict[str, Any]], TokenTriple]] = None,
    span_kind: str = "LLM",
) -> Callable[[str], Dict[str, Any]]:
    """
    Enrobe un pipeline RAG `query -> {answer, contexts, ...}` pour tracer la
    performance (latence + tokens) dans Phoenix. Si le SDK OpenTelemetry est
    absent, renvoie le pipeline INCHANGÉ (dégradation propre, jamais bloquant).
    """
    try:
        provider, tracer = _build_tracer(project_name, endpoint)
    except ImportError as e:
        logger.error(
            "SDK OpenTelemetry absent (%s) — pipeline non tracé. "
            "pip install -r requirements-phoenix.txt", e
        )
        return pipeline
    except Exception as e:
        logger.error("Init tracer échouée (%s) — pipeline non tracé.", e)
        return pipeline

    extract = token_extractor or _default_token_extractor

    calls: list = []   # historique (query, span_id, result) pour rattacher les scores

    def _wrapped(query: str) -> Dict[str, Any]:
        with tracer.start_as_current_span("rag_pipeline") as span:
            span.set_attribute(ATTR_SPAN_KIND, span_kind)
            span.set_attribute(ATTR_INPUT, str(query))
            span.set_attribute("input.mime_type", "text/plain")
            try:
                result = pipeline(query)   # la latence = durée de ce bloc
            except Exception as e:
                span.record_exception(e)
                raise
            if isinstance(result, dict):
                span.set_attribute(ATTR_OUTPUT, str(result.get("answer", "")))
                span.set_attribute("output.mime_type", "text/plain")
                for j, ctx in enumerate(result.get("contexts", []) or []):
                    span.set_attribute(
                        f"retrieval.documents.{j}.document.content", str(ctx)
                    )
                p, c, t = extract(result)
                if p is not None:
                    span.set_attribute(ATTR_TOK_PROMPT, int(p))
                if c is not None:
                    span.set_attribute(ATTR_TOK_COMPLETION, int(c))
                if t is not None:
                    span.set_attribute(ATTR_TOK_TOTAL, int(t))
            span_id_hex = format(span.get_span_context().span_id, "016x")
        calls.append({"query": query, "span_id": span_id_hex, "result": result})
        return result

    # Attributs exposés : flush + historique des appels (pour rattacher les scores).
    _wrapped.flush = provider.force_flush  # type: ignore[attr-defined]
    _wrapped.calls = calls                 # type: ignore[attr-defined]
    return _wrapped
