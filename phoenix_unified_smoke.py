"""
phoenix_unified_smoke.py — UNE seule trace par cas, avec TOUT réuni :
latence + tokens + réponse (performance)  ET  scores de cohérence (qualité).

C'est le bon usage de Phoenix : on trace l'exécution réelle du pipeline, puis on
rattache les scores d'évaluation AUX MÊMES traces (via leur span_id). Résultat :
un seul projet 'rageval' où chaque cas montre à la fois sa perf et sa qualité.

Prérequis :
  1. Serveur Phoenix lancé : docker compose up -d
  2. Client installé       : pip install -r requirements-phoenix.txt
                             (opentelemetry-* + arize-phoenix-client)

Lancer :  python phoenix_unified_smoke.py
Puis ouvrir : http://localhost:6006  ->  projet 'rageval'
"""
import random
import time

from phoenix_trace import traced_pipeline
from phoenix_export import log_span_scores

PROJET = "rageval"   # UN seul projet pour perf + qualité


def faux_pipeline(q: str) -> dict:
    """Pipeline factice : latence simulée + tokens (façon OpenAI)."""
    time.sleep(0.3)
    return {
        "answer": f"réponse à : {q}",
        "contexts": ["Loi 89-462, art. 15 : congé pour vente, préavis de six mois."],
        "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
    }


if __name__ == "__main__":
    # 1. On trace le pipeline (perf) dans le projet 'rageval'.
    pipe = traced_pipeline(faux_pipeline, project_name=PROJET)

    questions = [
        "Quel préavis pour un congé pour vente ?",
        "Le dépôt de garantie est-il plafonné ?",
        "Qui paie les réparations locatives ?",
    ]
    for q in questions:
        pipe(q)
    if hasattr(pipe, "flush"):
        pipe.flush()

    calls = getattr(pipe, "calls", [])
    if not calls:
        print("⚠️ Pipeline non tracé (OpenTelemetry absent ?). "
              "pip install -r requirements-phoenix.txt")
        raise SystemExit(1)

    # 2. On calcule les scores. EN VRAI : agent.evaluate_ragas(...) / evaluate_llm_judge(...)
    #    sur les réponses de `calls`. ICI (démo) : scores synthétiques par appel.
    span_scores = []
    for c in calls:
        span_scores.append((c["span_id"], {
            "faithfulness": round(random.uniform(0.55, 0.95), 2),
            "answer_relevancy": round(random.uniform(0.60, 0.95), 2),
            "context_precision": round(random.uniform(0.50, 0.90), 2),
            "context_recall": round(random.uniform(0.50, 0.90), 2),
        }))

    # 3. On laisse Phoenix ingérer les spans, puis on rattache les scores AUX MÊMES traces.
    time.sleep(2)
    n = log_span_scores(span_scores, annotator_kind="CODE")

    print(f"\n✅ {len(calls)} trace(s) (latence + tokens + réponse) "
          f"+ {n} mesure(s) sur LES MÊMES traces.")
    print(f"   http://localhost:6006 -> projet '{PROJET}' : perf ET qualité réunies.")
    if not n:
        print("   (0 mesure : installe arize-phoenix-client, ou augmente le sleep.)")
