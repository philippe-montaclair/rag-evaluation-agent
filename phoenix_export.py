"""
phoenix_export.py — Envoi des résultats d'évaluation RagEval vers Phoenix (Arize)
==================================================================================
Objectif : VISUALISER la qualité du RAG dans l'UI Phoenix (http://localhost:6006),
AVEC les scores affichés comme *mesures* (annotations d'évaluation), pas seulement
comme attributs de span.

Comment ça marche (en 2 temps) :
  1. On émet UNE trace OpenTelemetry par cas évalué (question=input, réponse=output,
     contextes récupérés) → visible dans Phoenix.
  2. On rattache les scores (faithfulness, answer_relevancy, juge…) à chaque span
     via l'API d'annotations du client Phoenix → ils apparaissent comme MESURES,
     filtrables et triables dans l'UI.

Montage (voir docker-compose.yml) :
  - Serveur Phoenix dans Docker (UI 6006, ingest OTLP gRPC 4317).
  - Côté hôte, uniquement des libs clientes légères (py3.9+, dont 3.13).

Dépendances : `pip install -r requirements-phoenix.txt`
    - opentelemetry-sdk + exporter OTLP gRPC   -> émission des traces
    - arize-phoenix-client                     -> annotations (les mesures)

Usage :
    from rag_evaluation_agent import RagEval
    from phoenix_export import export_run_to_phoenix
    run = agent.evaluate_ragas(dataset, version_id="v1")
    export_run_to_phoenix(run, project_name="rag-locatif")
    # -> http://localhost:6006
"""
from __future__ import annotations

import logging
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("RagEval.phoenix")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

OTLP_ENDPOINT_DEFAULT = "http://localhost:4317"   # ingest OTLP gRPC (traces)
PHOENIX_UI_DEFAULT = "http://localhost:6006"        # UI + REST (annotations)

# Conventions d'attributs OpenInference (en clair = pas de dépendance dure).
ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_INPUT = "input.value"
ATTR_OUTPUT = "output.value"
ATTR_PROJECT = "openinference.project.name"

_META_KEYS = {"id", "question", "answer", "contexts", "ground_truth", "justification"}


def _wait_for_phoenix(ui_url: str = PHOENIX_UI_DEFAULT, timeout_s: float = 30.0) -> bool:
    """Attend que le serveur Phoenix réponde (évite la course au démarrage)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(ui_url, timeout=2) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            time.sleep(1.0)
    logger.warning("Phoenix ne répond pas sur %s après %ss.", ui_url, timeout_s)
    return False


def _build_tracer(project_name: str, endpoint: str):
    """TracerProvider synchrone (SimpleSpanProcessor) exportant vers Phoenix.

    SimpleSpanProcessor = export immédiat et déterministe (petits volumes) :
    on sait tout de suite si l'émission échoue, plutôt qu'un envoi asynchrone
    « optimiste ». Lève ImportError si le SDK OpenTelemetry est absent.
    """
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )

    resource = Resource.create(
        {"service.name": project_name, ATTR_PROJECT: project_name}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(
        SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    return provider, provider.get_tracer("rageval")


def _emit_spans(run, project_name, endpoint, span_kind) -> List[Tuple[str, Dict[str, float]]]:
    """Émet une trace par cas ; retourne [(span_id_hex, {métrique: score}), ...]."""
    provider, tracer = _build_tracer(project_name, endpoint)
    scores = run.get("scores") or []
    metric_keys = [
        k for k in scores[0].keys()
        if k not in _META_KEYS and not str(k).startswith("_")
    ]
    out: List[Tuple[str, Dict[str, float]]] = []
    for i, case in enumerate(scores):
        case_id = case.get("id", f"case_{i}")
        case_scores: Dict[str, float] = {}
        with tracer.start_as_current_span(f"rag_eval:{case_id}") as span:
            span.set_attribute(ATTR_SPAN_KIND, span_kind)
            span.set_attribute(ATTR_INPUT, str(case.get("question", "")))
            span.set_attribute("input.mime_type", "text/plain")
            span.set_attribute(ATTR_OUTPUT, str(case.get("answer", "")))
            span.set_attribute("output.mime_type", "text/plain")
            span.set_attribute("eval.run_id", str(run.get("id", "")))
            span.set_attribute("eval.level", str(run.get("type", "")))
            for j, ctx in enumerate(case.get("contexts", []) or []):
                span.set_attribute(
                    f"retrieval.documents.{j}.document.content", str(ctx)
                )
            for mk in metric_keys:
                v = case.get(mk)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    span.set_attribute(f"eval.{mk}.score", float(v))
                    case_scores[mk] = float(v)
            span_id_hex = format(span.get_span_context().span_id, "016x")
        out.append((span_id_hex, case_scores))
    provider.force_flush()
    return out


def _log_annotations(spans, run, ui_url) -> int:
    """Attache les scores aux spans comme annotations (les 'mesures' de l'UI).

    Retourne le nombre d'annotations effectivement enregistrées. Nécessite
    `arize-phoenix-client`. RAGAS/TruLens -> annotator 'CODE' ; juge -> 'LLM'.
    """
    try:
        from phoenix.client import Client
    except ImportError:
        logger.warning(
            "arize-phoenix-client absent : traces envoyées mais SANS mesures. "
            "Fais : pip install arize-phoenix-client"
        )
        return 0

    annotator = "LLM" if run.get("type") == "llm_judge" else "CODE"
    return log_span_scores(spans, annotator_kind=annotator, ui_url=ui_url)


def log_span_scores(
    span_scores: List[Tuple[str, Dict[str, float]]],
    annotator_kind: str = "CODE",
    ui_url: str = PHOENIX_UI_DEFAULT,
) -> int:
    """
    Attache des scores à des spans EXISTANTS (par span_id), comme annotations.
    `span_scores` = [(span_id_hex, {métrique: score}), ...]. Réutilisable pour
    rattacher des scores à des traces produites par le pipeline (phoenix_trace).
    Nécessite `arize-phoenix-client`. Retourne le nombre d'annotations logguées.
    """
    try:
        from phoenix.client import Client
    except ImportError:
        logger.warning(
            "arize-phoenix-client absent : scores NON rattachés. "
            "pip install arize-phoenix-client"
        )
        return 0

    payload: List[Dict[str, Any]] = []
    for span_id_hex, case_scores in span_scores:
        for metric, score in case_scores.items():
            payload.append({
                "name": metric,
                "span_id": span_id_hex,
                "annotator_kind": annotator_kind,
                "result": {"score": float(score)},
            })
    if not payload:
        return 0

    client = Client(base_url=ui_url)
    for attempt in range(3):
        try:
            client.spans.log_span_annotations(span_annotations=payload)
            return len(payload)
        except Exception as e:
            if attempt == 2:
                logger.error("Échec du log des annotations : %s", e)
                return 0
            time.sleep(1.5)
    return 0


def export_run_to_phoenix(
    run: Dict[str, Any],
    project_name: str = "rageval",
    endpoint: str = OTLP_ENDPOINT_DEFAULT,
    ui_url: str = PHOENIX_UI_DEFAULT,
    span_kind: str = "CHAIN",
    wait: bool = True,
) -> Dict[str, int]:
    """
    Envoie un run d'évaluation vers Phoenix : 1 trace/cas + les scores en mesures.

    Retourne {"spans": n_traces, "annotations": n_mesures}. Ne lève jamais.
    """
    result = {"spans": 0, "annotations": 0}
    scores = run.get("scores") or []
    if not scores:
        logger.warning("Run sans détail par cas ('scores' vide) — rien à envoyer.")
        return result

    if wait:
        _wait_for_phoenix(ui_url)

    try:
        spans = _emit_spans(run, project_name, endpoint, span_kind)
    except ImportError as e:
        logger.error(
            "SDK OpenTelemetry absent (%s). pip install -r requirements-phoenix.txt", e
        )
        return result
    except Exception as e:
        logger.error("Émission des traces échouée : %s", e)
        return result

    result["spans"] = len(spans)
    # Laisse à Phoenix le temps d'ingérer les spans avant d'y rattacher les mesures.
    time.sleep(2.0)
    result["annotations"] = _log_annotations(spans, run, ui_url)

    logger.info(
        "Phoenix : %d trace(s) + %d mesure(s) — projet '%s'. UI : %s",
        result["spans"], result["annotations"], project_name, ui_url,
    )
    return result


def export_history_to_phoenix(
    agent,
    project_name: str = "rageval",
    endpoint: str = OTLP_ENDPOINT_DEFAULT,
    ui_url: str = PHOENIX_UI_DEFAULT,
) -> Dict[str, int]:
    """Envoie TOUS les runs de l'historique. Retourne les totaux cumulés."""
    total = {"spans": 0, "annotations": 0}
    for run in getattr(agent, "history", []):
        r = export_run_to_phoenix(
            run, project_name=project_name, endpoint=endpoint, ui_url=ui_url
        )
        total["spans"] += r["spans"]
        total["annotations"] += r["annotations"]
    return total
