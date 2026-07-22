"""
phoenix_perf_smoke.py — test de la trace de PERFORMANCE (latence + tokens).

Prérequis :
  1. Serveur Phoenix lancé   : docker compose up -d
  2. Client OTel installé    : pip install -r requirements-phoenix.txt

Lancer :  python phoenix_perf_smoke.py
Puis ouvrir : http://localhost:6006  ->  projet 'rageval-perf'
"""
import time

from phoenix_trace import traced_pipeline


def faux_pipeline(q: str) -> dict:
    """Pipeline factice : simule un peu de latence + expose des tokens (façon OpenAI)."""
    time.sleep(0.3)  # ~300 ms de latence simulée
    return {
        "answer": f"réponse à : {q}",
        "contexts": ["Loi 89-462, art. 15 : congé pour vente, préavis de six mois."],
        "usage": {"prompt_tokens": 120, "completion_tokens": 45, "total_tokens": 165},
    }


if __name__ == "__main__":
    pipe = traced_pipeline(faux_pipeline, project_name="rageval-perf")

    questions = [
        "Quel préavis pour un congé pour vente ?",
        "Le dépôt de garantie est-il plafonné ?",
        "Qui paie les réparations locatives ?",
    ]
    for question in questions:
        out = pipe(question)
        print(f"• {question}  ->  {out['answer'][:40]}…")

    # Vide le buffer pour être sûr que tout part avant la fin du script.
    if hasattr(pipe, "flush"):
        pipe.flush()

    print(
        "\n✅ 3 appels tracés. Ouvre http://localhost:6006 -> projet 'rageval-perf'."
        "\n   Chaque trace montre sa latence (~0,3 s) et ses tokens (prompt/completion/total)."
    )
