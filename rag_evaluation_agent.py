"""
═══════════════════════════════════════════════════════════════════
RAGEvaluationAgent — Agent d'évaluation automatique pour pipelines RAG
═══════════════════════════════════════════════════════════════════

OBJECTIF :
Automatiser l'évaluation d'un pipeline RAG (Retrieval-Augmented
Generation) AVANT la vérification humaine finale, afin de PRIORISER
le travail humain — jamais de le remplacer.

SCHÉMA DES 3 NIVEAUX D'ÉVALUATION :

    NIVEAU 1 — Métriques automatiques (RAGAS / TruLens)
        → rapide, scalable, filtre grossier
    NIVEAU 2 — LLM-as-Judge (évaluation qualitative fine)
        → plus coûteux, capture nuances (ton, clarté, métier)
    NIVEAU 3 — Vérification humaine
        → lent, coûteux, mais seul niveau réellement fiable
        → décision finale sur les cas signalés par les niveaux 1 et 2

    L'agent ne fait JAMAIS l'impasse sur le niveau 3. Il réduit
    seulement le volume de travail humain nécessaire.

MÉTHODES PRINCIPALES :
    evaluate_ragas()              -> scores RAGAS (faithfulness, etc.)
    evaluate_trulens()             -> suivi continu optionnel (TruLens)
    llm_as_judge()                  -> notation qualitative fine par LLM
    flag_for_human_review()         -> sélectionne les cas à vérifier
    generate_human_review_sheet()   -> fiches de vérification humaine
    integrate_human_feedback()      -> réinjecte la décision humaine
    compare_versions()              -> diff entre 2 versions du pipeline
    get_trend()                     -> évolution d'une métrique dans le temps
    generate_dashboard()            -> dashboard HTML (Chart.js)
    export()                        -> export JSON / CSV / Markdown
    summary()                       -> résumé exécutif + recommandations
    export_to_prometheus()          -> export optionnel vers Prometheus
═══════════════════════════════════════════════════════════════════
"""

import json
import csv
import io
import logging
import random
import statistics
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────
# Logger propre (pas de print())
# ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("RagEval")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────
# Détection des librairies optionnelles (fallback propre, jamais de crash)
# ─────────────────────────────────────────────────────────────────
try:
    import ragas  # noqa: F401
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    )
    from ragas import evaluate as ragas_evaluate
    from datasets import Dataset

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    # Fallbacks pour que le module reste importable ET patchable (tests, CI)
    faithfulness = answer_relevancy = context_precision = None
    context_recall = answer_correctness = None
    ragas_evaluate = None
    Dataset = None
    logger.warning("RAGAS non installé : `pip install ragas datasets` pour l'activer.")

try:
    import trulens_eval  # noqa: F401

    TRULENS_AVAILABLE = True
except ImportError:
    TRULENS_AVAILABLE = False
    logger.info("TruLens non installé (optionnel) : `pip install trulens_eval`.")

try:
    import phoenix as px  # noqa: F401

    PHOENIX_AVAILABLE = True
except ImportError:
    PHOENIX_AVAILABLE = False
    logger.info("Phoenix non installé (optionnel) : `pip install arize-phoenix`.")

try:
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CollectorRegistry = Gauge = push_to_gateway = None
    logger.info("prometheus_client non installé (optionnel).")

# Client LLM générique — adapte selon ton fournisseur (OpenAI ici en exemple)
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None
    logger.warning("openai non installé : nécessaire pour llm_as_judge() par défaut.")


# ─────────────────────────────────────────────────────────────────
# Configuration par défaut (externalisable / surchargeable)
# ─────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "ragas_metrics": [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
    ],
    "judge_criteria": ["exactitude", "clarte", "ton", "completude"],
    "human_review": {
        "grey_zone_tolerance": 0.10,   # ± autour du seuil
        "critical_threshold": 0.5,     # score en dessous = signalement direct
        "random_sample_pct": 0.05,     # 5% des bons scores tirés au hasard
        "score_threshold": 0.7,        # seuil de référence "bon score"
    },
}


JUDGE_SCALE_MIN = 1
JUDGE_SCALE_MAX = 5


def _normalize_judge_score(value: float) -> float:
    """Convertit une note d'échelle 1-5 vers 0-1 (bornée). 5->1.0, 1->0.0."""
    span = JUDGE_SCALE_MAX - JUDGE_SCALE_MIN
    if span <= 0:
        return float(value)
    return max(0.0, min(1.0, (value - JUDGE_SCALE_MIN) / span))


class RagEval:
    """
    Agent d'évaluation automatique pour pipeline RAG.

    Le pipeline évalué doit respecter l'interface :
        pipeline(query: str) -> {
            "answer": str,
            "contexts": List[str],
            "ground_truth": Optional[str]
        }
    """

    def __init__(
        self,
        pipeline: Callable[[str], Dict[str, Any]],
        config: Optional[Dict[str, Any]] = None,
        judge_model: str = "gpt-4o-mini",
        judge_provider: str = "openai",
        judge_base_url: Optional[str] = None,
        ragas_provider: str = "openai",
        ragas_base_url: Optional[str] = None,
        ragas_embedding_model: str = "nomic-embed-text",
        ragas_model: Optional[str] = None,
        ragas_timeout: int = 300,
        ragas_max_workers: int = 1,
        pipeline_model: Optional[str] = None,
        enable_trulens: bool = False,
        enable_phoenix: bool = False,
    ):
        self.pipeline = pipeline
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.judge_model = judge_model
        self.pipeline_model = pipeline_model or "unknown"

        self.enable_trulens = enable_trulens and TRULENS_AVAILABLE
        self.enable_phoenix = enable_phoenix and PHOENIX_AVAILABLE

        if enable_trulens and not TRULENS_AVAILABLE:
            logger.warning("enable_trulens=True mais TruLens absent : ignoré.")
        if enable_phoenix and not PHOENIX_AVAILABLE:
            logger.warning("enable_phoenix=True mais Phoenix absent : ignoré.")

        self.judge_provider = judge_provider
        self.judge_base_url = judge_base_url
        # Backend RAGAS (niveau 1) :
        #  - "openai" : comportement par défaut de RAGAS (clé OpenAI requise) ;
        #  - "ollama" : LLM + embeddings 100% locaux injectés dans ragas.evaluate().
        self.ragas_provider = ragas_provider
        self.ragas_base_url = ragas_base_url            # ex. "http://localhost:11434"
        self.ragas_embedding_model = ragas_embedding_model
        # Modèle LLM pour RAGAS (peut différer du juge : privilégier un modèle
        # *instruct* non "thinking" pour un JSON propre). None -> reprend judge_model.
        self.ragas_model = ragas_model or judge_model
        # RAGAS sur backend LOCAL lent : timeout large + exécution quasi-série
        # (Ollama traite un modèle en série ; trop de workers = vague de TimeoutError).
        self.ragas_timeout = ragas_timeout
        self.ragas_max_workers = ragas_max_workers
        self._ragas_backend = None                      # (llm, embeddings) construit à la demande
        # Client LLM juge :
        #  - "openai" : client OpenAI classique ;
        #  - "ollama" : local, via _call_judge_ollama (package `ollama`), pas de client OpenAI requis.
        self._openai_client = None
        if judge_provider == "openai" and OPENAI_AVAILABLE:
            try:
                self._openai_client = OpenAI()
            except Exception as e:  # ex. clé API absente : non bloquant
                logger.warning(f"Client OpenAI non initialisé : {e}")

        # Historique : liste de "runs" horodatés {id, timestamp, results, meta}
        self.history: List[Dict[str, Any]] = []
        # Cas signalés pour vérification humaine, avec décisions intégrées
        self.human_reviews: Dict[str, Dict[str, Any]] = {}

        if self.enable_phoenix:
            try:
                px.launch_app()
                logger.info("Session Phoenix lancée pour le tracing.")
            except Exception as e:
                logger.warning(f"Impossible de lancer Phoenix : {e}")

    # ═══════════════════════════════════════════════════════════
    # 3. ÉVALUATION AUTOMATIQUE — RAGAS
    # ═══════════════════════════════════════════════════════════
    def _build_ragas_ollama_backend(self):
        """
        Construit (et met en cache) le duo (LLM, embeddings) 100% local pour
        RAGAS, via langchain-ollama + les wrappers RAGAS. Aucun appel à une API
        externe : tout passe par le serveur Ollama local.

        Prérequis Mac : `ollama serve`, `ollama pull <judge_model>` et
        `ollama pull <ragas_embedding_model>` (ex. nomic-embed-text), plus
        `pip install langchain-ollama` (en plus de ragas/datasets).

        Lève une RuntimeError explicite si une dépendance manque :
        evaluate_ragas l'intercepte et renvoie {"error": ...} sans planter.
        """
        if self._ragas_backend is not None:
            return self._ragas_backend
        try:
            from langchain_ollama import ChatOllama, OllamaEmbeddings
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
        except ImportError as e:
            raise RuntimeError(
                "Backend RAGAS Ollama indisponible : "
                "`pip install langchain-ollama ragas datasets`."
            ) from e

        base_url = self.ragas_base_url or "http://localhost:11434"
        try:
            # reasoning=False : coupe le <think> des modèles type qwen3 qui casse
            # le JSON attendu par RAGAS (dispo sur langchain-ollama récent).
            chat = ChatOllama(
                model=self.ragas_model, base_url=base_url,
                temperature=0.0, reasoning=False,
            )
        except TypeError:
            chat = ChatOllama(model=self.ragas_model, base_url=base_url, temperature=0.0)
        emb = OllamaEmbeddings(model=self.ragas_embedding_model, base_url=base_url)
        self._ragas_backend = (
            LangchainLLMWrapper(chat),
            LangchainEmbeddingsWrapper(emb),
        )
        logger.info(
            f"Backend RAGAS Ollama prêt (llm={self.ragas_model}, "
            f"emb={self.ragas_embedding_model}, url={base_url})."
        )
        return self._ragas_backend

    def _ragas_run_config(self):
        """
        RunConfig adapté à un backend LOCAL lent (Ollama) : gros timeout et
        exécution quasi-série (max_workers) pour éviter la vague de TimeoutError
        quand RAGAS lance ses jobs en parallèle sur un seul serveur Ollama.
        Renvoie None si ragas n'est pas installé (dégradation propre).
        """
        try:
            from ragas.run_config import RunConfig
        except ImportError:
            return None
        return RunConfig(timeout=self.ragas_timeout, max_workers=self.ragas_max_workers)

    def evaluate_ragas(
        self, dataset: List[Dict[str, Any]], version_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcule les métriques RAGAS sur un dataset de questions.

        Chaque métrique mesure :
        - faithfulness        : la réponse est-elle fidèle aux contextes
                                 récupérés (pas d'hallucination) ?
        - answer_relevancy    : la réponse répond-elle bien à la question
                                 posée (pas hors-sujet) ?
        - context_precision   : les documents récupérés sont-ils pertinents
                                 (peu de bruit) ?
        - context_recall      : a-t-on récupéré tout ce qui était
                                 nécessaire pour répondre correctement ?
        - answer_correctness  : la réponse est-elle correcte par rapport
                                 à une vérité de référence (ground_truth) ?
                                 -> nécessite ground_truth, sinon ignorée.

        dataset : liste de dicts avec les clés
            question, answer, contexts, ground_truth (optionnel)
        """
        if not dataset:
            logger.error("evaluate_ragas: dataset vide, évaluation annulée.")
            return {"scores": [], "averages": {}, "error": "dataset vide"}

        if not RAGAS_AVAILABLE:
            logger.error("RAGAS non installé : évaluation impossible.")
            return {"scores": [], "averages": {}, "error": "ragas non installé"}

        has_ground_truth = all(d.get("ground_truth") for d in dataset)
        if not has_ground_truth:
            logger.warning(
                "ground_truth manquant pour au moins une question : "
                "answer_correctness sera ignorée."
            )

        metrics_map = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }
        selected_metrics = [
            metrics_map[m]
            for m in self.config["ragas_metrics"]
            if m in metrics_map and (m != "answer_correctness" or has_ground_truth)
        ]

        try:
            hf_dataset = Dataset.from_list(
                [
                    {
                        "question": d["question"],
                        "answer": d["answer"],
                        "contexts": d["contexts"],
                        "ground_truth": d.get("ground_truth", ""),
                    }
                    for d in dataset
                ]
            )
            eval_kwargs: Dict[str, Any] = {"metrics": selected_metrics}
            if self.ragas_provider == "ollama":
                # RAGAS 100% local : on injecte LLM + embeddings Ollama.
                llm, embeddings = self._build_ragas_ollama_backend()
                eval_kwargs["llm"] = llm
                eval_kwargs["embeddings"] = embeddings
                run_config = self._ragas_run_config()
                if run_config is not None:
                    eval_kwargs["run_config"] = run_config
            result = ragas_evaluate(hf_dataset, **eval_kwargs)
            result_df = result.to_pandas()
        except Exception as e:
            logger.error(f"Erreur pendant l'évaluation RAGAS : {e}")
            return {"scores": [], "averages": {}, "error": str(e)}

        per_question = result_df.to_dict(orient="records")
        # On ne moyenne QUE les colonnes numériques : le schéma de sortie de
        # RAGAS varie selon les versions (question/answer/contexts vs
        # user_input/response/retrieved_contexts). Les colonnes texte lèvent
        # TypeError sur .mean() -> on les ignore proprement.
        averages: Dict[str, float] = {}
        for col in result_df.columns:
            try:
                moyenne = float(result_df[col].mean())
            except (TypeError, ValueError):
                continue
            if moyenne == moyenne:          # écarte les NaN (métrique 100% échouée)
                averages[col] = moyenne

        run = {
            "id": version_id or f"run_{len(self.history)+1}",
            "timestamp": datetime.now().isoformat(),
            "type": "ragas",
            "pipeline_model": self.pipeline_model,
            "scores": per_question,
            "averages": averages,
        }
        self.history.append(run)
        logger.info(f"Évaluation RAGAS terminée — moyennes : {averages}")
        return run

    # ═══════════════════════════════════════════════════════════
    # 4. SUIVI CONTINU — TRULENS (optionnel)
    # ═══════════════════════════════════════════════════════════
    def evaluate_trulens(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Évaluation via TruLens (feedback functions : groundedness,
        relevance, coherence). Ne s'exécute que si enable_trulens=True
        et la librairie est disponible. Sinon, retourne un résultat vide
        sans bloquer le reste du pipeline.
        """
        if not self.enable_trulens:
            logger.info("evaluate_trulens ignoré (désactivé ou indisponible).")
            return {"scores": [], "averages": {}, "skipped": True}

        try:
            # Exemple minimal — à adapter selon la version de trulens_eval
            from trulens_eval import Feedback, TruBasicApp
            from trulens_eval.feedback.provider import OpenAI as TruOpenAI

            provider = TruOpenAI()
            f_groundedness = Feedback(provider.groundedness_measure_with_cot_reasons)
            f_relevance = Feedback(provider.relevance)

            scores = []
            for item in dataset:
                context_str = " ".join(item["contexts"])
                groundedness = f_groundedness(context_str, item["answer"])
                relevance = f_relevance(item["question"], item["answer"])
                scores.append(
                    {
                        "question": item["question"],
                        "groundedness": groundedness,
                        "relevance": relevance,
                    }
                )

            averages = {
                "groundedness": statistics.mean(s["groundedness"] for s in scores),
                "relevance": statistics.mean(s["relevance"] for s in scores),
            }
            run = {
                "id": f"trulens_run_{len(self.history)+1}",
                "timestamp": datetime.now().isoformat(),
                "type": "trulens",
                "scores": scores,
                "averages": averages,
            }
            self.history.append(run)
            return run
        except Exception as e:
            logger.warning(f"Erreur TruLens (non bloquante) : {e}")
            return {"scores": [], "averages": {}, "error": str(e)}

    # ═══════════════════════════════════════════════════════════
    # 5. ÉVALUATION QUALITATIVE FINE — LLM-AS-JUDGE
    # ═══════════════════════════════════════════════════════════
    def llm_as_judge(
        self,
        question: str,
        context: str,
        answer: str,
        criteria: Optional[List[str]] = None,
        double_pass: bool = False,
    ) -> Dict[str, Any]:
        """
        Fait noter une réponse par un LLM "juge" sur des critères
        métier personnalisables (1 à 5 par critère + justification).

        Le modèle juge (judge_model) peut être différent du modèle
        évalué (pipeline_model) pour limiter le biais d'auto-évaluation.

        double_pass=True : fait 2 appels et moyenne les scores pour
        réduire le bruit stochastique du LLM.
        """
        criteria = criteria or self.config["judge_criteria"]

        if self.judge_provider == "openai" and not self._openai_client:
            logger.error("Client LLM indisponible : llm_as_judge annulé.")
            return {"error": "client LLM non configuré"}

        prompt = self._build_judge_prompt(question, context, answer, criteria)

        passes = 2 if double_pass else 1
        all_results = []
        for i in range(passes):
            result = self._call_judge(prompt)
            if result:
                all_results.append(result)

        if not all_results:
            logger.error("llm_as_judge : aucune réponse exploitable du juge.")
            return {crit: None for crit in criteria} | {
                "justification": "Erreur : réponse du juge non exploitable."
            }

        # Moyenne des scores (échelle brute 1-5) PUIS normalisation 0-1
        # -> cohérent avec RAGAS/TruLens et avec les seuils de flag_for_human_review.
        averaged: Dict[str, Any] = {}
        raw_avg: Dict[str, Any] = {}
        for crit in criteria:
            values = [r.get(crit) for r in all_results if isinstance(r.get(crit), (int, float))]
            if values:
                mean_raw = statistics.mean(values)
                raw_avg[crit] = round(mean_raw, 2)
                averaged[crit] = round(_normalize_judge_score(mean_raw), 3)
            else:
                raw_avg[crit] = None
                averaged[crit] = None

        averaged["justification"] = all_results[0].get("justification", "")
        averaged["_scale"] = "0-1 (normalisé depuis 1-5)"
        averaged["_raw_1_5"] = raw_avg
        return averaged

    def evaluate_llm_judge(
        self,
        dataset: List[Dict[str, Any]],
        criteria: Optional[List[str]] = None,
        version_id: Optional[str] = None,
        double_pass: bool = False,
    ) -> Dict[str, Any]:
        """
        Applique llm_as_judge à un jeu de cas ET enregistre un run dans
        l'historique (type "llm_judge", niveau 2) — pour que le juge
        apparaisse dans le dashboard, les tendances et le résumé, au même
        titre que RAGAS. Scores stockés en 0-1 (normalisés).

        dataset : liste de dicts {question, contexts|context, answer}.
        """
        criteria = criteria or self.config["judge_criteria"]
        if not dataset:
            return {"scores": [], "averages": {}, "error": "dataset vide"}
        if self.judge_provider == "openai" and not self._openai_client:
            return {"scores": [], "averages": {}, "error": "client LLM non configuré"}

        per_item: List[Dict[str, Any]] = []
        for i, item in enumerate(dataset):
            context = item.get("context")
            if context is None:
                context = " ".join(item.get("contexts", []))
            judged = self.llm_as_judge(
                item.get("question", ""), context, item.get("answer", ""),
                criteria=criteria, double_pass=double_pass,
            )
            row = {"id": item.get("id", f"case_{i}"), "question": item.get("question", "")}
            for crit in criteria:
                row[crit] = judged.get(crit)
            per_item.append(row)

        averages: Dict[str, float] = {}
        for crit in criteria:
            vals = [r[crit] for r in per_item if isinstance(r.get(crit), (int, float))]
            if vals:
                averages[crit] = round(statistics.mean(vals), 3)

        run = {
            "id": version_id or f"judge_run_{len(self.history) + 1}",
            "timestamp": datetime.now().isoformat(),
            "type": "llm_judge",
            "judge_model": self.judge_model,
            "scores": per_item,
            "averages": averages,
        }
        self.history.append(run)
        logger.info(f"Évaluation LLM-juge terminée — moyennes (0-1) : {averages}")
        return run

    def _build_judge_prompt(
        self, question: str, context: str, answer: str, criteria: List[str]
    ) -> str:
        criteria_desc = ", ".join(criteria)
        return f"""Tu es un évaluateur impartial de qualité de réponses générées par un système RAG.

Question posée : {question}

Contexte fourni au système : {context}

Réponse générée : {answer}

Évalue la réponse selon ces critères (note de 1 à 5 chacun) : {criteria_desc}.

Réponds STRICTEMENT en JSON valide, sans texte autour, au format :
{{
{", ".join(f'"{c}": <note 1-5>' for c in criteria)},
"justification": "<courte justification en une ou deux phrases>"
}}"""

    def _call_judge(self, prompt: str) -> Optional[Dict[str, Any]]:
        if self.judge_provider == "ollama":
            return self._call_judge_ollama(prompt)
        try:
            response = self._openai_client.chat.completions.create(
                model=self.judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            raw = response.choices[0].message.content.strip()
            # Nettoyage basique si le modèle entoure le JSON de ```
            raw = raw.strip("`").replace("json\n", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Réponse du juge non parsable en JSON, tentative ignorée.")
            return None
        except Exception as e:
            logger.error(f"Erreur d'appel au LLM juge : {e}")
            return None

    def _call_judge_ollama(self, prompt: str) -> Optional[Dict[str, Any]]:
        """
        Juge 100% LOCAL via Ollama (aucune donnée envoyée à une API externe).
        Utilise `format="json"` et `think=False` (modèles « thinking » type qwen3 :
        évite que le raisonnement pollue le JSON). Dégrade proprement si le package
        `ollama` est absent ou si la réponse n'est pas du JSON exploitable.
        """
        try:
            import ollama  # import différé : le module reste utilisable sans ollama
        except ImportError:
            logger.error("Package `ollama` absent : `pip install ollama`.")
            return None

        kwargs: Dict[str, Any] = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "format": "json",
            "options": {"temperature": 0.0},
        }
        try:
            try:
                resp = ollama.chat(think=False, **kwargs)  # think=False si supporté
            except TypeError:
                resp = ollama.chat(**kwargs)               # anciennes versions d'ollama
            raw = (resp["message"]["content"] or "").strip()
            raw = raw.strip("`").replace("json\n", "").strip()
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Réponse du juge Ollama non parsable en JSON, ignorée.")
            return None
        except Exception as e:
            logger.error(f"Erreur d'appel au juge Ollama : {e}")
            return None

    # ═══════════════════════════════════════════════════════════
    # 6. SÉLECTION POUR VÉRIFICATION HUMAINE
    # ═══════════════════════════════════════════════════════════
    def flag_for_human_review(
        self, results: List[Dict[str, Any]], score_key: str = "faithfulness"
    ) -> List[Dict[str, Any]]:
        """
        Sélectionne automatiquement les cas à faire vérifier par un
        humain selon 3 critères combinés :
          a) zone grise (proche du seuil de référence)
          b) score sous le seuil critique
          c) échantillon aléatoire même sur bons scores (contrôle qualité)

        results : liste de dicts contenant au moins `score_key` et
        idéalement question/answer/contexts pour la fiche de review.
        """
        cfg = self.config["human_review"]
        threshold = cfg["score_threshold"]
        tolerance = cfg["grey_zone_tolerance"]
        critical = cfg["critical_threshold"]
        sample_pct = cfg["random_sample_pct"]

        flagged = []
        good_scores_pool = []

        for i, r in enumerate(results):
            score = r.get(score_key)
            if score is None:
                continue

            case_id = r.get("id", f"case_{i}")
            if score < critical:
                flagged.append({**r, "id": case_id, "reason": "seuil_critique"})
            elif abs(score - threshold) <= tolerance:
                flagged.append({**r, "id": case_id, "reason": "zone_grise"})
            else:
                good_scores_pool.append({**r, "id": case_id})

        # Échantillon aléatoire sur les bons scores
        n_sample = max(1, int(len(good_scores_pool) * sample_pct)) if good_scores_pool else 0
        sample = random.sample(good_scores_pool, min(n_sample, len(good_scores_pool)))
        for s in sample:
            flagged.append({**s, "reason": "echantillon_aleatoire"})

        logger.info(
            f"{len(flagged)} cas signalés pour vérification humaine "
            f"(sur {len(results)} au total)."
        )
        return flagged

    def generate_human_review_sheet(
        self, flagged_cases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Génère une fiche structurée par cas signalé, prête à être
        exportée et remplie par un humain.
        """
        sheets = []
        for case in flagged_cases:
            sheet = {
                "id": case.get("id"),
                "question": case.get("question", ""),
                "answer": case.get("answer", ""),
                "scores_ragas": {
                    k: v for k, v in case.items()
                    if k in self.config["ragas_metrics"]
                },
                "scores_llm_judge": case.get("llm_judge_scores", {}),
                "raison_signalement": case.get("reason", ""),
                "decision": None,  # à remplir : "valide" | "corrige" | "rejete"
                "commentaire": "",
            }
            sheets.append(sheet)
            self.human_reviews[sheet["id"]] = sheet
        return sheets

    def integrate_human_feedback(
        self, case_id: str, decision: str, comment: str = ""
    ) -> bool:
        """
        Réinjecte la décision humaine dans l'historique (boucle de
        feedback fermée). decision doit être "valide", "corrige" ou "rejete".
        """
        if decision not in ("valide", "corrige", "rejete"):
            logger.error(f"Décision invalide : {decision}")
            return False

        if case_id not in self.human_reviews:
            logger.warning(f"Cas {case_id} introuvable dans les fiches de review.")
            return False

        self.human_reviews[case_id]["decision"] = decision
        self.human_reviews[case_id]["commentaire"] = comment
        self.human_reviews[case_id]["reviewed_at"] = datetime.now().isoformat()
        logger.info(f"Feedback humain intégré pour {case_id} : {decision}")
        return True

    # ═══════════════════════════════════════════════════════════
    # 7. HISTORIQUE ET COMPARAISON DE VERSIONS
    # ═══════════════════════════════════════════════════════════
    def compare_versions(self, version_a_id: str, version_b_id: str) -> Dict[str, Any]:
        """
        Compare les scores moyens de deux runs de l'historique.
        Retourne un diff métrique par métrique avec le sens de variation.
        """
        run_a = next((r for r in self.history if r["id"] == version_a_id), None)
        run_b = next((r for r in self.history if r["id"] == version_b_id), None)

        if not run_a or not run_b:
            logger.error("compare_versions : une des deux versions est introuvable.")
            return {"error": "version introuvable"}

        diff = {}
        for metric in set(run_a["averages"]) | set(run_b["averages"]):
            a_val = run_a["averages"].get(metric)
            b_val = run_b["averages"].get(metric)
            if a_val is None or b_val is None:
                diff[metric] = {"a": a_val, "b": b_val, "delta": None}
                continue
            delta = b_val - a_val
            pct = (delta / a_val * 100) if a_val != 0 else None
            diff[metric] = {
                "a": round(a_val, 3),
                "b": round(b_val, 3),
                "delta": round(delta, 3),
                "pct_change": round(pct, 1) if pct is not None else None,
                "trend": "amélioration" if delta > 0 else ("régression" if delta < 0 else "stable"),
            }
        return diff

    def get_trend(self, metric_name: str) -> List[Tuple[str, float]]:
        """
        Retourne l'évolution d'une métrique donnée dans le temps,
        sous forme de liste (timestamp, valeur).
        """
        trend = []
        for run in self.history:
            val = run.get("averages", {}).get(metric_name)
            if val is not None:
                trend.append((run["timestamp"], val))
        return trend

    # ═══════════════════════════════════════════════════════════
    # 8. VISUALISATION — DASHBOARD
    # ═══════════════════════════════════════════════════════════
    def generate_dashboard(self, output_path: str = "dashboard.html") -> str:
        """
        Génère un dashboard HTML autonome (CDN Chart.js) qui superpose
        les trois niveaux d'évaluation :

          • Niveau 1 (automatique)  : radar RAGAS + barres TruLens ;
          • Niveau 1-bis (évolution): courbe multi-métriques dans le temps
            + comparaison des deux dernières versions (détection de régression) ;
          • Niveau 3 (humain)       : table des cas signalés pour relecture.

        Ne dépend d'aucun réseau à l'exécution du HTML (Chart.js via CDN).
        Robuste à un historique vide.
        """
        if not self.history:
            logger.warning("Aucun historique disponible pour le dashboard.")

        ragas_runs = [r for r in self.history if r.get("type") == "ragas"]
        trulens_runs = [r for r in self.history if r.get("type") == "trulens"]

        # --- Niveau 1 : radar de la dernière évaluation automatique ---
        last_auto = (ragas_runs or trulens_runs or [{"averages": {}}])[-1]
        averages = last_auto.get("averages", {})
        radar_labels = list(averages.keys())
        radar_values = [round(v, 3) for v in averages.values()]

        # --- Niveau 1-bis : évolution des métriques RAGAS dans le temps ---
        trend_labels = [r.get("timestamp", r.get("id", "")) for r in ragas_runs]
        metric_keys: List[str] = []
        for r in ragas_runs:
            for k in r.get("averages", {}):
                if k not in metric_keys:
                    metric_keys.append(k)
        palette = [
            "rgb(54,162,235)", "rgb(255,99,132)", "rgb(75,192,192)",
            "rgb(255,159,64)", "rgb(153,102,255)", "rgb(201,203,207)",
        ]
        trend_datasets = [
            {
                "label": mk,
                "data": [
                    (round(r["averages"][mk], 3) if mk in r.get("averages", {}) else None)
                    for r in ragas_runs
                ],
                "borderColor": palette[i % len(palette)],
                "fill": False,
                "spanGaps": True,
            }
            for i, mk in enumerate(metric_keys)
        ]

        # --- Comparaison des deux dernières versions ---
        compare_labels: List[str] = []
        compare_a: List[float] = []
        compare_b: List[float] = []
        compare_id_a = compare_id_b = ""
        if len(ragas_runs) >= 2:
            run_a, run_b = ragas_runs[-2], ragas_runs[-1]
            compare_id_a, compare_id_b = run_a.get("id", "A"), run_b.get("id", "B")
            for mk in metric_keys:
                compare_labels.append(mk)
                compare_a.append(round(run_a.get("averages", {}).get(mk, 0.0), 3))
                compare_b.append(round(run_b.get("averages", {}).get(mk, 0.0), 3))

        # --- Niveau 1-bis : TruLens (si présent) ---
        tl_avg = (trulens_runs[-1].get("averages", {}) if trulens_runs else {})
        trulens_labels = list(tl_avg.keys())
        trulens_values = [round(v, 3) for v in tl_avg.values()]

        # --- Niveau 2 : LLM-as-Judge (si présent) ---
        judge_runs = [r for r in self.history if r.get("type") == "llm_judge"]
        judge_avg = (judge_runs[-1].get("averages", {}) if judge_runs else {})
        judge_labels = list(judge_avg.keys())
        judge_values = [round(v, 3) for v in judge_avg.values()]

        # --- Niveau 3 : table des cas signalés pour relecture humaine ---
        flagged_table_rows = "".join(
            f"<tr><td>{c.get('id')}</td><td>{str(c.get('question',''))[:60]}</td>"
            f"<td>{c.get('reason','')}</td></tr>"
            for c in self.human_reviews.values()
        ) or "<tr><td colspan='3'><em>Aucun cas signalé.</em></td></tr>"

        has_compare = "block" if len(ragas_runs) >= 2 else "none"
        has_trulens = "block" if trulens_labels else "none"
        has_judge = "block" if judge_labels else "none"

        html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="UTF-8">
<title>Dashboard Évaluation RAG</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{ font-family: sans-serif; margin: 2em; background: #f8f9fa; color: #222; }}
h1, h2 {{ color: #333; }}
.levels {{ display:flex; gap:1em; flex-wrap:wrap; margin:1em 0; }}
.level {{ flex:1; min-width:220px; padding:0.8em 1em; border-radius:8px; background:#fff;
         border-left:5px solid #36a2eb; }}
.level.n2 {{ border-left-color:#ffb020; }}
.level.n3 {{ border-left-color:#f96; }}
.level small {{ color:#666; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1em; background:#fff; }}
td, th {{ border: 1px solid #ccc; padding: 6px; text-align: left; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 2em; }}
.card {{ background:#fff; padding:1em; border-radius:8px; }}
canvas {{ max-width: 480px; }}
</style>
</head>
<body>
<h1>📊 Dashboard d'évaluation RAG</h1>
<p>Dernière évaluation automatique : {last_auto.get('timestamp', 'N/A')}
   — {len(ragas_runs)} run(s) RAGAS, {len(trulens_runs)} run(s) TruLens.</p>

<div class="levels">
  <div class="level"><b>Niveau 1 — Automatique</b><br>
    <small>RAGAS / TruLens. Rapide, scalable, filtre grossier.</small></div>
  <div class="level n2"><b>Niveau 2 — LLM-as-Judge</b><br>
    <small>Notation qualitative fine (0-1, normalisée depuis 1-5).</small></div>
  <div class="level n3"><b>Niveau 3 — Humain</b><br>
    <small>Seul niveau fiable. Cas priorisés ci-dessous pour relecture.</small></div>
</div>

<div class="charts">
  <div class="card"><h2>Niveau 1 — Scores moyens (radar)</h2><canvas id="radarChart"></canvas></div>
  <div class="card" style="display:{has_judge}"><h2>Niveau 2 — LLM-as-Judge</h2><canvas id="judgeChart"></canvas></div>
  <div class="card"><h2>Évolution dans le temps</h2><canvas id="trendChart"></canvas></div>
  <div class="card" style="display:{has_compare}"><h2>Comparaison {compare_id_a} → {compare_id_b}</h2><canvas id="compareChart"></canvas></div>
  <div class="card" style="display:{has_trulens}"><h2>TruLens (groundedness / relevance)</h2><canvas id="trulensChart"></canvas></div>
</div>

<h2>🔎 Cas signalés pour vérification humaine (niveau 3)</h2>
<table>
<tr><th>ID</th><th>Question</th><th>Raison</th></tr>
{flagged_table_rows}
</table>

<script>
const R = {{ scales: {{ r: {{ suggestedMin: 0, suggestedMax: 1 }} }} }};
const Y01 = {{ scales: {{ y: {{ suggestedMin: 0, suggestedMax: 1 }} }} }};

new Chart(document.getElementById('radarChart'), {{
  type: 'radar',
  data: {{ labels: {json.dumps(radar_labels)}, datasets: [{{
    label: 'Score moyen', data: {json.dumps(radar_values)}, fill: true,
    backgroundColor: 'rgba(54,162,235,0.2)', borderColor: 'rgb(54,162,235)' }}] }},
  options: R
}});

new Chart(document.getElementById('trendChart'), {{
  type: 'line',
  data: {{ labels: {json.dumps(trend_labels)}, datasets: {json.dumps(trend_datasets)} }},
  options: Y01
}});

if (document.getElementById('compareChart') && {json.dumps(bool(compare_labels))}) {{
  new Chart(document.getElementById('compareChart'), {{
    type: 'bar',
    data: {{ labels: {json.dumps(compare_labels)}, datasets: [
      {{ label: {json.dumps(compare_id_a)}, data: {json.dumps(compare_a)}, backgroundColor:'rgba(201,203,207,0.7)' }},
      {{ label: {json.dumps(compare_id_b)}, data: {json.dumps(compare_b)}, backgroundColor:'rgba(54,162,235,0.7)' }}
    ] }},
    options: Y01
  }});
}}

if (document.getElementById('trulensChart') && {json.dumps(bool(trulens_labels))}) {{
  new Chart(document.getElementById('trulensChart'), {{
    type: 'bar',
    data: {{ labels: {json.dumps(trulens_labels)}, datasets: [{{
      label: 'TruLens', data: {json.dumps(trulens_values)}, backgroundColor:'rgba(255,159,64,0.7)' }}] }},
    options: Y01
  }});
}}

if (document.getElementById('judgeChart') && {json.dumps(bool(judge_labels))}) {{
  new Chart(document.getElementById('judgeChart'), {{
    type: 'bar',
    data: {{ labels: {json.dumps(judge_labels)}, datasets: [{{
      label: 'LLM-as-Judge (0-1)', data: {json.dumps(judge_values)}, backgroundColor:'rgba(255,176,32,0.7)' }}] }},
    options: Y01
  }});
}}
</script>

</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"Dashboard généré : {output_path}")
        return output_path


    # ═══════════════════════════════════════════════════════════
    # 9. EXPORT DES RÉSULTATS
    # ═══════════════════════════════════════════════════════════
    def export(
        self, format: str = "json", filepath: Optional[str] = None
    ) -> str:
        """
        Exporte l'ensemble des résultats (historique, cas signalés,
        décisions humaines) dans le format demandé.
        """
        format = format.lower()
        if format == "json":
            content = self._export_json()
        elif format == "csv":
            content = self._export_csv()
        elif format == "markdown":
            content = self._export_markdown()
        else:
            raise ValueError(f"Format non supporté : {format}")

        if filepath:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            logger.info(f"Export {format} écrit dans {filepath}")

        return content

    def _export_json(self) -> str:
        data = {
            "history": self.history,
            "human_reviews": self.human_reviews,
            "exported_at": datetime.now().isoformat(),
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def _export_csv(self) -> str:
        output = io.StringIO()
        if not self.history or not self.history[-1].get("scores"):
            return ""

        last_scores = self.history[-1]["scores"]
        fieldnames = sorted({k for row in last_scores for k in row.keys()})
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in last_scores:
            writer.writerow(row)
        return output.getvalue()

    def _export_markdown(self) -> str:
        lines = ["# Rapport d'évaluation RAG\n"]
        if self.history:
            last = self.history[-1]
            lines.append(f"## Dernière évaluation ({last['timestamp']})\n")
            lines.append("| Métrique | Score moyen |")
            lines.append("|---|---|")
            for k, v in last.get("averages", {}).items():
                lines.append(f"| {k} | {v:.3f} |")
            lines.append("")

        if len(self.history) >= 2:
            lines.append("## Comparaison des 2 dernières versions\n")
            diff = self.compare_versions(
                self.history[-2]["id"], self.history[-1]["id"]
            )
            lines.append("| Métrique | Avant | Après | Delta | Tendance |")
            lines.append("|---|---|---|---|---|")
            for metric, d in diff.items():
                lines.append(
                    f"| {metric} | {d.get('a')} | {d.get('b')} | "
                    f"{d.get('delta')} | {d.get('trend')} |"
                )
            lines.append("")

        if self.human_reviews:
            lines.append("## Fiches de vérification humaine\n")
            for case_id, sheet in self.human_reviews.items():
                lines.append(f"### Cas {case_id}")
                lines.append(f"- **Question** : {sheet['question']}")
                lines.append(f"- **Réponse** : {sheet['answer']}")
                lines.append(f"- **Raison du signalement** : {sheet['raison_signalement']}")
                lines.append(f"- **Décision** : {sheet.get('decision') or '_à faire_'}")
                lines.append(f"- **Commentaire** : {sheet.get('commentaire') or '-'}")
                lines.append("")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 10. RÉSUMÉ EXÉCUTIF
    # ═══════════════════════════════════════════════════════════
    def summary(self) -> str:
        """
        Résumé lisible en quelques lignes : scores principaux,
        nombre de cas signalés, tendance, recommandations.
        """
        if not self.history:
            return "Aucune évaluation disponible."

        last = self.history[-1]
        averages = last.get("averages", {})
        n_flagged = len(self.human_reviews)

        lines = [f"📋 Résumé exécutif — run {last['id']} ({last['timestamp']})", ""]
        for k, v in averages.items():
            lines.append(f"  - {k}: {v:.3f}")

        lines.append(f"\n👥 Cas signalés pour vérification humaine : {n_flagged}")

        if len(self.history) >= 2:
            diff = self.compare_versions(self.history[-2]["id"], last["id"])
            regressions = [m for m, d in diff.items() if d.get("trend") == "régression"]
            if regressions:
                lines.append(f"⚠️ Régression détectée sur : {', '.join(regressions)}")
            else:
                lines.append("✅ Pas de régression détectée par rapport à la version précédente.")

        # Recommandations simples basées sur les scores les plus faibles
        if averages:
            worst = sorted(averages.items(), key=lambda x: x[1])[:2]
            lines.append("\n💡 Recommandations :")
            for metric, val in worst:
                if val < 0.6:
                    lines.append(f"  - Améliorer '{metric}' (score faible : {val:.2f})")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════
    # 11. INTÉGRATIONS OPTIONNELLES
    # ═══════════════════════════════════════════════════════════
    def export_to_prometheus(
        self, pushgateway_url: str = "localhost:9091", job_name: str = "rag_eval"
    ) -> bool:
        """
        Pousse les métriques moyennes de la dernière évaluation vers
        un Prometheus Pushgateway. Ne fait rien si prometheus_client
        n'est pas installé.
        """
        if not PROMETHEUS_AVAILABLE:
            logger.info("prometheus_client absent : export ignoré.")
            return False

        if not self.history:
            logger.warning("Aucun historique à exporter vers Prometheus.")
            return False

        registry = CollectorRegistry()
        averages = self.history[-1].get("averages", {})
        for metric, value in averages.items():
            g = Gauge(f"rag_{metric}", f"Score moyen RAGAS pour {metric}", registry=registry)
            g.set(value)

        try:
            push_to_gateway(pushgateway_url, job=job_name, registry=registry)
            logger.info(f"Métriques poussées vers Prometheus ({pushgateway_url}).")
            return True
        except Exception as e:
            logger.error(f"Erreur d'export Prometheus : {e}")
            return False

    def export_to_phoenix(
        self,
        run: Optional[Dict[str, Any]] = None,
        project_name: str = "rageval",
    ) -> Dict[str, int]:
        """
        Envoie les scores vers Phoenix (Arize) pour VISUALISER la qualité du RAG
        dans l'UI (http://localhost:6006) : une trace par cas + les scores comme
        MESURES (annotations). Si `run` est None, envoie tout l'historique.
        Import paresseux. Retourne {"spans": n, "annotations": n}.

        Prérequis : serveur Phoenix lancé (docker-compose.yml) et
        `pip install -r requirements-phoenix.txt`.
        """
        try:
            from phoenix_export import (
                export_run_to_phoenix,
                export_history_to_phoenix,
            )
        except ImportError as e:
            logger.error(f"phoenix_export indisponible : {e}")
            return {"spans": 0, "annotations": 0}

        if run is not None:
            return export_run_to_phoenix(run, project_name=project_name)
        return export_history_to_phoenix(self, project_name=project_name)


# ═══════════════════════════════════════════════════════════════════
# ALIAS DE COMPATIBILITÉ — l'ancien nom reste importable
# ═══════════════════════════════════════════════════════════════════
RAGEvaluationAgent = RagEval  # rétro-compat : ne pas retirer avant une version majeure


# ═══════════════════════════════════════════════════════════════════
# EXEMPLE D'UTILISATION COMPLET
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # 1. Pipeline factice respectant l'interface attendue
    def fake_pipeline(query: str) -> Dict[str, Any]:
        return {
            "answer": f"Réponse simulée pour : {query}",
            "contexts": ["Contexte simulé A", "Contexte simulé B"],
            "ground_truth": "Réponse de référence simulée",
        }

    agent = RagEval(
        pipeline=fake_pipeline,
        judge_model="gpt-4o-mini",
        pipeline_model="mon-rag-v1",
        enable_trulens=False,
        enable_phoenix=False,
    )

    # Dataset de test
    test_dataset = [
        {
            "question": "Quelle est la capitale de la France ?",
            "answer": "Paris est la capitale de la France.",
            "contexts": ["Paris est la capitale de la France depuis..."],
            "ground_truth": "Paris",
        },
        {
            "question": "Qui a écrit Les Misérables ?",
            "answer": "Victor Hugo a écrit Les Misérables.",
            "contexts": ["Victor Hugo, écrivain français..."],
            "ground_truth": "Victor Hugo",
        },
    ]

    # 2. Évaluation RAGAS simple
    run_v1 = agent.evaluate_ragas(test_dataset, version_id="v1")

    # 3. Évaluation LLM-as-Judge sur le premier cas
    judge_result = agent.llm_as_judge(
        question=test_dataset[0]["question"],
        context=test_dataset[0]["contexts"][0],
        answer=test_dataset[0]["answer"],
        double_pass=True,
    )
    print("Résultat LLM-as-Judge :", judge_result)

    # 4. Sélection des cas à vérifier par un humain
    flagged = agent.flag_for_human_review(run_v1["scores"], score_key="faithfulness")

    # 5. Génération des fiches de vérification humaine
    sheets = agent.generate_human_review_sheet(flagged)

    # Simulation d'une décision humaine
    if sheets:
        agent.integrate_human_feedback(sheets[0]["id"], "valide", "RAS, conforme.")

    # 6. Comparaison entre deux versions (simulation d'une v2)
    run_v2 = agent.evaluate_ragas(test_dataset, version_id="v2")
    diff = agent.compare_versions("v1", "v2")
    print("Diff v1 vs v2 :", diff)

    # 7. Génération du dashboard HTML
    agent.generate_dashboard("dashboard.html")

    # 8. Export dans les 3 formats
    agent.export(format="json", filepath="rapport.json")
    agent.export(format="csv", filepath="rapport.csv")
    agent.export(format="markdown", filepath="rapport.md")

    # 9. Résumé exécutif
    print(agent.summary())