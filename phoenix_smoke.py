"""
phoenix_smoke.py — test de plomberie Phoenix (sans RAGAS ni OpenAI).

But : vérifier que l'export OTel -> Phoenix fonctionne, en envoyant un run
SYNTHÉTIQUE (scores inventés). Si tu vois les traces dans l'UI, le tuyau est bon
et il ne reste plus qu'à brancher de vrais scores (evaluate_ragas / evaluate_llm_judge).

Prérequis :
  1. Serveur Phoenix lancé :  docker compose up -d
  2. Client OTel installé   :  pip install -r requirements-phoenix.txt

Lancer :  python phoenix_smoke.py
Puis ouvrir : http://localhost:6006
"""
from phoenix_export import export_run_to_phoenix

# Un "run" a la même forme que ce que renvoient evaluate_ragas / evaluate_llm_judge.
run_synthetique = {
    "id": "smoke-v1",
    "timestamp": "2026-07-20T23:59:00",
    "type": "ragas",
    "scores": [
        {
            "id": "cas_1",
            "question": "Quel préavis pour un congé pour vente ?",
            "answer": "Le bailleur doit respecter un préavis de 6 mois avant l'échéance.",
            "contexts": ["Loi 89-462, art. 15 : congé pour vente, préavis de six mois."],
            "faithfulness": 0.82,
            "answer_relevancy": 0.78,
            "context_precision": 0.66,
            "context_recall": 0.71,
        },
        {
            "id": "cas_2",
            "question": "Le dépôt de garantie est-il plafonné ?",
            "answer": "Oui, un mois de loyer hors charges pour une location vide.",
            "contexts": ["Loi 89-462, art. 22 : dépôt de garantie limité à un mois de loyer."],
            "faithfulness": 0.55,        # volontairement bas -> à repérer dans l'UI
            "answer_relevancy": 0.80,
            "context_precision": 0.60,
            "context_recall": 0.62,
        },
    ],
    "averages": {"faithfulness": 0.685, "answer_relevancy": 0.79},
}

if __name__ == "__main__":
    res = export_run_to_phoenix(run_synthetique, project_name="rageval-smoke")
    if res["spans"]:
        print(f"\n✅ {res['spans']} trace(s) + {res['annotations']} mesure(s) envoyées.")
        print("   Ouvre http://localhost:6006 -> projet 'rageval-smoke'.")
        if not res["annotations"]:
            print("   (Traces OK mais 0 mesure : installe arize-phoenix-client pour voir les scores.)")
    else:
        print("\n⚠️ Rien envoyé : Phoenix tourne-t-il ? Libs OTel installées ?")
