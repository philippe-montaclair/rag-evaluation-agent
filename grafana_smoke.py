"""
grafana_smoke.py — alimente le dashboard Grafana avec des métriques synthétiques.

Pousse des scores (qualité) + latence/tokens (perf) vers le Pushgateway, que
Prometheus scrape et que Grafana affiche. Sert à voir le dashboard « clignoter »
sans avoir besoin d'un vrai pipeline.

Prérequis :
  1. Stack lancée : cd monitoring && docker compose -f docker-compose.grafana.yml up -d
  2. Client       : pip install prometheus_client

Lancer :  python grafana_smoke.py
Puis ouvrir : http://localhost:3001  (admin/admin) -> dashboard « RagEval — Qualité & Performance »
"""
import random
import time

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

PUSHGATEWAY = "localhost:9091"
JOB = "rageval"

# Mêmes noms que ceux poussés par RagEval.export_to_prometheus() : rag_<métrique>.
METRICS = {
    "rag_faithfulness": "Fidélité (0-1)",
    "rag_answer_relevancy": "Pertinence de la réponse (0-1)",
    "rag_context_precision": "Précision du contexte (0-1)",
    "rag_context_recall": "Rappel du contexte (0-1)",
    "rag_latency_ms": "Latence du dernier appel (ms)",
    "rag_tokens_total": "Tokens (total)",
}


def push_once() -> dict:
    registry = CollectorRegistry()
    gauges = {name: Gauge(name, desc, registry=registry) for name, desc in METRICS.items()}

    values = {
        "rag_faithfulness": round(random.uniform(0.55, 0.95), 3),
        "rag_answer_relevancy": round(random.uniform(0.60, 0.95), 3),
        "rag_context_precision": round(random.uniform(0.50, 0.90), 3),
        "rag_context_recall": round(random.uniform(0.50, 0.90), 3),
        "rag_latency_ms": round(random.uniform(250, 900), 1),
        "rag_tokens_total": random.randint(120, 320),
    }
    for name, g in gauges.items():
        g.set(values[name])

    push_to_gateway(PUSHGATEWAY, job=JOB, registry=registry)
    return values


if __name__ == "__main__":
    print(f"Envoi de métriques vers {PUSHGATEWAY} (job={JOB})…")
    for i in range(12):                     # ~12 points -> les courbes bougent
        v = push_once()
        print(f"  [{i+1:>2}/12] faithfulness={v['rag_faithfulness']}  "
              f"latence={v['rag_latency_ms']}ms  tokens={v['rag_tokens_total']}")
        time.sleep(5)                        # 5 s entre chaque push
    print("\n✅ Terminé. Ouvre http://localhost:3001 (admin/admin) -> "
          "dashboard « RagEval — Qualité & Performance ».")
