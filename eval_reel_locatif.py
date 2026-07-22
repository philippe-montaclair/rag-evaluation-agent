"""
eval_reel_locatif.py — VRAIE évaluation du RAG droit locatif (100% local).

Contrairement aux scripts *_smoke.py (données synthétiques), celui-ci évalue le
VRAI pipeline de agents-gestion-locative :
  1. chercher_loi(question, k=4)      -> contextes réellement récupérés ;
  2. synthèse d'une réponse via Ollama (qwen3:8b) à partir de ces contextes ;
  3. juge LLM local (RagEval.evaluate_llm_judge, judge_provider="ollama") sur 3
     critères -> scores 0-1 réels, historisés (niveau 2) ;
  4. export vers Phoenix (traces + mesures) et, en option, Prometheus/Grafana.

Aucune donnée n'est envoyée à une API externe (tout tourne sur Ollama en local).

Prérequis (côté Mac) :
  - `ollama serve` lancé, modèle présent : `ollama pull qwen3:8b`
  - index Chroma construit : `python scripts/build_index.py` (dans agents-gestion-locative)
  - `pip install ollama` (+ requirements-phoenix.txt si export Phoenix)
  - serveur Phoenix lancé (docker compose up -d) si on exporte

Lancer :  python eval_reel_locatif.py --n 5        # 5 premières questions
          python eval_reel_locatif.py --n 34       # jeu complet (lent)
          python eval_reel_locatif.py --n 5 --no-phoenix
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- Chemins vers le projet frère agents-gestion-locative ---
HERE = Path(__file__).resolve().parent                 # agent_evaluation_rag/
PROJECTS = HERE.parent                                 # dossier des projets
GESTION = PROJECTS / "agents-gestion-locative"
sys.path.insert(0, str(HERE))                                       # rag_evaluation_agent, phoenix_export
sys.path.insert(0, str(GESTION / "agent_impaye" / "outils"))        # chercher_loi
sys.path.insert(0, str(GESTION / "eval"))                           # eval_agent_impaye (JEU_EVAL), reformulation_requete

from rag_evaluation_agent import RagEval

MODELE_JUGE = "qwen3:8b"   # même modèle que ton éval aval (adapter si besoin : mistral:7b plus rapide)
CRITERES = ["fidelite_au_contexte", "pertinence_reponse", "absence_hallucination"]


def _synthetiser_reponse(question: str, contextes: list, modele: str = MODELE_JUGE) -> str:
    """Réponse courte à juger, générée UNIQUEMENT à partir des contextes (repris de ton éval aval)."""
    import ollama  # import différé
    contexte_str = "\n\n".join(f"- {c}" for c in contextes) or "(aucun contexte trouvé)"
    prompt = (
        "Tu réponds à une question de droit locatif en t'appuyant EXCLUSIVEMENT sur les "
        "extraits de loi ci-dessous.\n"
        "Règles impératives :\n"
        "- Réponds DIRECTEMENT à la question dès le premier mot : ne reformule pas la "
        "question, pas d'introduction ni de formule d'accroche.\n"
        "- N'affirme QUE ce qui figure explicitement dans les extraits ; n'ajoute aucun "
        "chiffre, délai, condition ou exception absent des extraits.\n"
        "- Si les extraits ne suffisent pas, écris exactement : "
        "« Les extraits ne permettent pas de répondre. »\n"
        "- 1 à 3 phrases factuelles, aucune digression, aucun commentaire.\n\n"
        f"Question : {question}\n\nExtraits de loi :\n{contexte_str}\n\n"
        "Réponse :"
    )
    resp = ollama.chat(
        model=modele,
        think=False,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.0, "num_predict": 400, "num_ctx": 8192},
    )
    return resp["message"]["content"].strip()


def construire_dataset(n: int) -> list:
    """Pour chaque question : contextes réels (chercher_loi) + réponse synthétisée + réponse-or."""
    from chercher_loi import chercher_loi
    from eval_agent_impaye import JEU_EVAL

    dataset = []
    for q in JEU_EVAL[:n]:
        contextes = [r["texte"] for r in chercher_loi(q["question"], k=4)]
        reponse = _synthetiser_reponse(q["question"], contextes)
        dataset.append({
            "id": q["id"],
            "question": q["question"],
            "contexts": contextes,
            "answer": reponse,
            "ground_truth": q.get("reponse_or", ""),
        })
        print(f"  · {q['id']} : {len(contextes)} contexte(s), réponse {len(reponse)} car.")
    return dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Éval réelle locale du RAG droit locatif (juge Ollama).")
    ap.add_argument("--n", type=int, default=5, help="nombre de questions (défaut 5)")
    ap.add_argument("--juge", default=MODELE_JUGE, help="modèle juge Ollama (défaut qwen3:8b)")
    ap.add_argument("--ragas", action="store_true",
                    help="lancer AUSSI RAGAS niveau 1 (100%% local via Ollama)")
    ap.add_argument("--ragas-emb", default="bge-m3",
                    help="modèle d'embeddings Ollama pour RAGAS (défaut bge-m3, multilingue ; nomic-embed-text sous-note le français)")
    ap.add_argument("--ragas-llm", default=None,
                    help="modèle LLM Ollama pour RAGAS (défaut = juge ; préférer un modèle instruct non-thinking)")
    ap.add_argument("--dump", action="store_true",
                    help="sauver le jeu construit (question/réponse/contextes) en JSON pour inspection")
    ap.add_argument("--ragas-metrics", default=None,
                    help="sous-ensemble RAGAS séparé par des virgules "
                         "(ex. 'faithfulness,answer_relevancy,context_precision,context_recall' pour écarter "
                         "answer_correctness, peu fiable avec un juge 7B). Défaut = set complet.")
    ap.add_argument("--no-phoenix", action="store_true", help="ne pas exporter vers Phoenix")
    ap.add_argument("--prometheus", action="store_true", help="pousser aussi vers Prometheus/Grafana")
    args = ap.parse_args()

    print(f"1) Construction du jeu ({args.n} questions) — chercher_loi + synthèse Ollama…")
    try:
        dataset = construire_dataset(args.n)
    except Exception as e:
        print(f"\n⚠️ Impossible de construire le jeu : {e}")
        print("   Vérifie : `ollama serve` lancé, index Chroma construit, `pip install ollama`.")
        sys.exit(1)

    if args.dump:
        import json
        from datetime import datetime
        f = HERE / f"dump_locatif_{datetime.now():%Y%m%d_%H%M%S}.json"
        f.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"   -> jeu sauvegardé : {f.name} (inspecter les réponses, ex. Q3)")

    print(f"\n2) Jugement local (juge={args.juge}, critères={CRITERES})…")
    agent_config = None
    if args.ragas_metrics:
        agent_config = {"ragas_metrics": [m.strip() for m in args.ragas_metrics.split(",") if m.strip()]}
    agent = RagEval(
        pipeline=lambda q: {"answer": "", "contexts": [], "ground_truth": ""},
        config=agent_config,
        judge_provider="ollama",
        judge_model=args.juge,
        ragas_provider="ollama",
        ragas_embedding_model=args.ragas_emb,
        ragas_model=args.ragas_llm,
        pipeline_model="agent_impaye/chercher_loi+synthese_eval",
    )
    run = agent.evaluate_llm_judge(dataset, criteria=CRITERES, version_id="locatif-v1")

    print("\n3) Scores moyens (0-1) :")
    for critere, valeur in run.get("averages", {}).items():
        print(f"   - {critere}: {valeur:.3f}")

    if args.ragas:
        print(f"\n2b) RAGAS niveau 1 (100%% local, emb={args.ragas_emb})…")
        print("     Prérequis : `ollama pull nomic-embed-text` + `pip install langchain-ollama ragas datasets`.")
        ragas_run = agent.evaluate_ragas(dataset, version_id="locatif-ragas-v1")
        if ragas_run.get("error"):
            print(f"     ⚠️ RAGAS ignoré : {ragas_run['error']}")
        else:
            for metrique, valeur in ragas_run.get("averages", {}).items():
                print(f"     - {metrique}: {valeur:.3f}")
            # Détail par question : repère un vrai point faible d'une simple sous-mesure du juge/embeddings.
            print("     Détail par question (faithfulness / answer_relevancy) :")
            for i, ligne in enumerate(ragas_run.get("scores", [])):
                fa = ligne.get("faithfulness")
                ar = ligne.get("answer_relevancy")
                qid = dataset[i]["id"] if i < len(dataset) else f"case_{i}"
                fa_s = f"{fa:.2f}" if isinstance(fa, (int, float)) else "—"
                ar_s = f"{ar:.2f}" if isinstance(ar, (int, float)) else "—"
                print(f"       · {qid} : faith={fa_s}  relev={ar_s}")

    if not args.no_phoenix:
        print("\n4) Export vers Phoenix (traces + mesures)…")
        try:
            res = agent.export_to_phoenix(run, project_name="rag-locatif")
            print(f"   -> {res}. UI : http://localhost:6006 (projet 'rag-locatif')")
        except Exception as e:
            print(f"   ⚠️ Export Phoenix ignoré : {e}")

    if args.prometheus:
        print("\n5) Export vers Prometheus/Grafana…")
        try:
            # NB : les métriques seront nommées rag_<critere> (fidelite_au_contexte…),
            # à refléter dans les panneaux Grafana (le dashboard fourni cible les noms RAGAS).
            agent.export_to_prometheus(pushgateway_url="localhost:9091", job_name="rageval")
            print("   -> poussé. Grafana : http://localhost:3001")
        except Exception as e:
            print(f"   ⚠️ Export Prometheus ignoré : {e}")

    print("\n✅ Éval réelle terminée.")


if __name__ == "__main__":
    main()
