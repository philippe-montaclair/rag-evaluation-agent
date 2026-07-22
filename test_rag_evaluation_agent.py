"""
═══════════════════════════════════════════════════════════════════
Tests unitaires — RAGEvaluationAgent
═══════════════════════════════════════════════════════════════════

Objectif : valider chaque méthode de l'agent indépendamment, en
mockant les dépendances externes (RAGAS, OpenAI, TruLens, Prometheus)
pour ne dépendre d'aucune clé API ni connexion réseau.

Lancer avec :
    pytest test_rag_evaluation_agent.py -v

Dépendances de test :
    pip install pytest pytest-mock
═══════════════════════════════════════════════════════════════════
"""

import json
import os
import statistics
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# On importe le module de l'agent (à adapter selon le nom réel du fichier)
import rag_evaluation_agent as rea
from rag_evaluation_agent import RAGEvaluationAgent, DEFAULT_CONFIG


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def fake_pipeline():
    def _pipeline(query: str):
        return {
            "answer": f"Réponse pour {query}",
            "contexts": ["contexte A", "contexte B"],
            "ground_truth": "vérité de référence",
        }
    return _pipeline


@pytest.fixture
def test_dataset():
    return [
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


@pytest.fixture
def agent(fake_pipeline):
    return RAGEvaluationAgent(
        pipeline=fake_pipeline,
        judge_model="gpt-4o-mini",
        pipeline_model="test-model",
        enable_trulens=False,
        enable_phoenix=False,
    )


@pytest.fixture
def agent_with_history(agent):
    """Agent pré-rempli avec deux runs factices dans l'historique."""
    agent.history = [
        {
            "id": "v1",
            "timestamp": datetime.now().isoformat(),
            "type": "ragas",
            "scores": [
                {"id": "case_0", "question": "Q1", "answer": "A1", "faithfulness": 0.9},
                {"id": "case_1", "question": "Q2", "answer": "A2", "faithfulness": 0.4},
            ],
            "averages": {"faithfulness": 0.65, "answer_relevancy": 0.8},
        },
        {
            "id": "v2",
            "timestamp": datetime.now().isoformat(),
            "type": "ragas",
            "scores": [
                {"id": "case_0", "question": "Q1", "answer": "A1", "faithfulness": 0.95},
                {"id": "case_1", "question": "Q2", "answer": "A2", "faithfulness": 0.3},
            ],
            "averages": {"faithfulness": 0.625, "answer_relevancy": 0.75},
        },
    ]
    return agent


# ═══════════════════════════════════════════════════════════════
# 1. TESTS — INITIALISATION
# ═══════════════════════════════════════════════════════════════

class TestInitialisation:

    def test_init_basic(self, fake_pipeline):
        agent = RAGEvaluationAgent(pipeline=fake_pipeline)
        assert agent.pipeline is fake_pipeline
        assert agent.config["ragas_metrics"] == DEFAULT_CONFIG["ragas_metrics"]
        assert agent.history == []
        assert agent.human_reviews == {}

    def test_init_custom_config_merge(self, fake_pipeline):
        custom = {"human_review": {"random_sample_pct": 0.2}}
        agent = RAGEvaluationAgent(pipeline=fake_pipeline, config=custom)
        assert agent.config["human_review"]["random_sample_pct"] == 0.2
        # les autres clés du config par défaut doivent rester présentes
        assert "ragas_metrics" in agent.config

    def test_init_trulens_disabled_if_unavailable(self, fake_pipeline):
        with patch.object(rea, "TRULENS_AVAILABLE", False):
            agent = RAGEvaluationAgent(pipeline=fake_pipeline, enable_trulens=True)
            assert agent.enable_trulens is False

    def test_init_phoenix_disabled_if_unavailable(self, fake_pipeline):
        with patch.object(rea, "PHOENIX_AVAILABLE", False):
            agent = RAGEvaluationAgent(pipeline=fake_pipeline, enable_phoenix=True)
            assert agent.enable_phoenix is False


# ═══════════════════════════════════════════════════════════════
# 2. TESTS — EVALUATE_RAGAS
# ═══════════════════════════════════════════════════════════════

class TestEvaluateRagas:

    def test_empty_dataset_returns_error(self, agent):
        result = agent.evaluate_ragas([])
        assert result["scores"] == []
        assert "error" in result

    def test_ragas_unavailable_returns_error(self, agent, test_dataset):
        with patch.object(rea, "RAGAS_AVAILABLE", False):
            result = agent.evaluate_ragas(test_dataset)
            assert "error" in result
            assert result["scores"] == []

    def test_ragas_success_appends_to_history(self, agent, test_dataset):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = [
            {"faithfulness": 0.9, "answer_relevancy": 0.8},
            {"faithfulness": 0.7, "answer_relevancy": 0.6},
        ]
        fake_df.columns = ["faithfulness", "answer_relevancy"]
        fake_df.__getitem__.side_effect = lambda col: MagicMock(
            mean=MagicMock(return_value=0.8 if col == "faithfulness" else 0.7)
        )

        fake_result = MagicMock()
        fake_result.to_pandas.return_value = fake_df

        with patch.object(rea, "RAGAS_AVAILABLE", True), \
             patch.object(rea, "ragas_evaluate", return_value=fake_result), \
             patch.object(rea, "Dataset") as mock_dataset:
            mock_dataset.from_list.return_value = MagicMock()
            result = agent.evaluate_ragas(test_dataset, version_id="test_run")

        assert result["id"] == "test_run"
        assert len(agent.history) == 1
        assert "faithfulness" in result["averages"]

    def test_ragas_missing_ground_truth_warns_but_continues(self, agent, caplog):
        dataset_no_gt = [
            {"question": "Q", "answer": "A", "contexts": ["C"], "ground_truth": ""}
        ]
        with patch.object(rea, "RAGAS_AVAILABLE", True), \
             patch.object(rea, "ragas_evaluate") as mock_eval, \
             patch.object(rea, "Dataset") as mock_dataset:
            fake_df = MagicMock()
            fake_df.to_dict.return_value = [{"faithfulness": 0.5}]
            fake_df.columns = ["faithfulness"]
            fake_df.__getitem__.side_effect = lambda col: MagicMock(mean=MagicMock(return_value=0.5))
            mock_eval.return_value.to_pandas.return_value = fake_df
            mock_dataset.from_list.return_value = MagicMock()

            with caplog.at_level("WARNING"):
                agent.evaluate_ragas(dataset_no_gt)

            assert any("ground_truth" in rec.message for rec in caplog.records)

    def test_ragas_exception_handled_gracefully(self, agent, test_dataset):
        with patch.object(rea, "RAGAS_AVAILABLE", True), \
             patch.object(rea, "Dataset") as mock_dataset:
            mock_dataset.from_list.side_effect = Exception("boom")
            result = agent.evaluate_ragas(test_dataset)
            assert "error" in result
            assert result["scores"] == []


# ═══════════════════════════════════════════════════════════════
# 2b. TESTS — RAGAS BACKEND OLLAMA (niveau 1 local)
# ═══════════════════════════════════════════════════════════════

class TestRagasOllamaBackend:

    def _fake_result(self):
        fake_df = MagicMock()
        fake_df.to_dict.return_value = [{"faithfulness": 0.9}]
        fake_df.columns = ["faithfulness"]
        fake_df.__getitem__.side_effect = lambda col: MagicMock(
            mean=MagicMock(return_value=0.9)
        )
        fake_result = MagicMock()
        fake_result.to_pandas.return_value = fake_df
        return fake_result

    def test_ollama_backend_forwards_llm_and_embeddings(self, fake_pipeline, test_dataset):
        """En provider ollama, ragas_evaluate reçoit llm= et embeddings= locaux."""
        agent = RAGEvaluationAgent(pipeline=fake_pipeline, ragas_provider="ollama")
        sentinel_llm, sentinel_emb = object(), object()
        with patch.object(rea, "RAGAS_AVAILABLE", True), \
             patch.object(rea, "ragas_evaluate", return_value=self._fake_result()) as mock_eval, \
             patch.object(rea, "Dataset") as mock_dataset, \
             patch.object(agent, "_build_ragas_ollama_backend",
                          return_value=(sentinel_llm, sentinel_emb)):
            mock_dataset.from_list.return_value = MagicMock()
            agent.evaluate_ragas(test_dataset)
        _, kwargs = mock_eval.call_args
        assert kwargs["llm"] is sentinel_llm
        assert kwargs["embeddings"] is sentinel_emb

    def test_openai_default_sends_no_backend_kwargs(self, fake_pipeline, test_dataset):
        """Le chemin openai par défaut reste inchangé (pas de llm/embeddings)."""
        agent = RAGEvaluationAgent(pipeline=fake_pipeline)  # défaut = openai
        with patch.object(rea, "RAGAS_AVAILABLE", True), \
             patch.object(rea, "ragas_evaluate", return_value=self._fake_result()) as mock_eval, \
             patch.object(rea, "Dataset") as mock_dataset:
            mock_dataset.from_list.return_value = MagicMock()
            agent.evaluate_ragas(test_dataset)
        _, kwargs = mock_eval.call_args
        assert "llm" not in kwargs and "embeddings" not in kwargs

    def test_ollama_backend_missing_dep_degrades_gracefully(self, fake_pipeline, test_dataset):
        """Sans langchain-ollama, evaluate_ragas renvoie {'error': ...} sans crash."""
        agent = RAGEvaluationAgent(pipeline=fake_pipeline, ragas_provider="ollama")
        with patch.object(rea, "RAGAS_AVAILABLE", True), \
             patch.object(rea, "Dataset") as mock_dataset:
            mock_dataset.from_list.return_value = MagicMock()
            result = agent.evaluate_ragas(test_dataset)
        assert "error" in result
        assert result["scores"] == []


# ═══════════════════════════════════════════════════════════════
# 3. TESTS — EVALUATE_TRULENS
# ═══════════════════════════════════════════════════════════════

class TestEvaluateTrulens:

    def test_skipped_if_disabled(self, agent, test_dataset):
        result = agent.evaluate_trulens(test_dataset)
        assert result["skipped"] is True

    def test_skipped_if_unavailable(self, fake_pipeline, test_dataset):
        with patch.object(rea, "TRULENS_AVAILABLE", False):
            agent = RAGEvaluationAgent(pipeline=fake_pipeline, enable_trulens=True)
            result = agent.evaluate_trulens(test_dataset)
            assert result["skipped"] is True


# ═══════════════════════════════════════════════════════════════
# 4. TESTS — LLM_AS_JUDGE
# ═══════════════════════════════════════════════════════════════

class TestLLMAsJudge:

    def test_no_client_returns_error(self, agent):
        agent._openai_client = None
        result = agent.llm_as_judge("Q", "C", "A")
        assert "error" in result

    def test_successful_single_pass(self, agent):
        fake_response = MagicMock()
        fake_response.choices[0].message.content = json.dumps({
            "exactitude": 5, "clarte": 4, "ton": 5, "completude": 4,
            "justification": "Bonne réponse."
        })
        agent._openai_client = MagicMock()
        agent._openai_client.chat.completions.create.return_value = fake_response

        result = agent.llm_as_judge("Q", "C", "A", double_pass=False)
        # scores normalisés 0-1 : 5 -> 1.0, 4 -> 0.75 (échelle 1-5)
        assert result["exactitude"] == 1.0
        assert result["clarte"] == 0.75
        assert result["_raw_1_5"]["exactitude"] == 5
        assert "justification" in result

    def test_double_pass_averages_scores(self, agent):
        responses = [
            json.dumps({"exactitude": 5, "clarte": 3, "ton": 4, "completude": 4, "justification": "ok"}),
            json.dumps({"exactitude": 3, "clarte": 5, "ton": 4, "completude": 4, "justification": "ok2"}),
        ]
        fake_calls = []
        for r in responses:
            resp = MagicMock()
            resp.choices[0].message.content = r
            fake_calls.append(resp)

        agent._openai_client = MagicMock()
        agent._openai_client.chat.completions.create.side_effect = fake_calls

        result = agent.llm_as_judge("Q", "C", "A", double_pass=True)
        # moyenne brute (5+3)/2 = 4 sur 1-5  ->  0.75 en 0-1
        assert result["exactitude"] == 0.75
        assert result["clarte"] == 0.75

    def test_malformed_json_handled(self, agent):
        fake_response = MagicMock()
        fake_response.choices[0].message.content = "pas du json valide"
        agent._openai_client = MagicMock()
        agent._openai_client.chat.completions.create.return_value = fake_response

        result = agent.llm_as_judge("Q", "C", "A")
        # aucune passe exploitable -> valeurs None + justification d'erreur
        assert result["exactitude"] is None
        assert "Erreur" in result["justification"]

    def test_api_exception_handled(self, agent):
        agent._openai_client = MagicMock()
        agent._openai_client.chat.completions.create.side_effect = Exception("timeout")

        result = agent.llm_as_judge("Q", "C", "A")
        assert result["exactitude"] is None

    def test_ollama_provider_no_openai_client_needed(self):
        # provider=ollama : aucun client OpenAI requis, routage vers _call_judge_ollama
        agent = RAGEvaluationAgent(
            pipeline=lambda q: {}, judge_model="mistral:7b", judge_provider="ollama"
        )
        assert agent._openai_client is None
        with patch.object(
            agent, "_call_judge_ollama",
            return_value={"exactitude": 4, "clarte": 4, "ton": 4,
                          "completude": 4, "justification": "ok"},
        ):
            r = agent.llm_as_judge("Q", "C", "A")
        assert r["exactitude"] == 0.75          # 4/5 normalisé en 0-1
        assert r["_raw_1_5"]["exactitude"] == 4


# ═══════════════════════════════════════════════════════════════
# 4-bis. TESTS — EVALUATE_LLM_JUDGE (historisation niveau 2)
# ═══════════════════════════════════════════════════════════════

class TestEvaluateLLMJudge:

    def _mock_judge(self, agent, notes):
        fake = MagicMock()
        fake.choices[0].message.content = json.dumps(
            {**notes, "justification": "ok"}
        )
        agent._openai_client = MagicMock()
        agent._openai_client.chat.completions.create.return_value = fake

    def test_records_run_in_history(self, agent):
        self._mock_judge(agent, {"exactitude": 4, "clarte": 4, "ton": 4, "completude": 4})
        dataset = [
            {"question": "Q1", "contexts": ["c"], "answer": "A1"},
            {"question": "Q2", "contexts": ["c"], "answer": "A2"},
        ]
        run = agent.evaluate_llm_judge(dataset, version_id="j1")
        assert run["type"] == "llm_judge"
        assert run["id"] == "j1"
        assert len(agent.history) == 1
        # 4 sur 1-5 -> 0.75 normalisé, moyenné sur les 2 items
        assert run["averages"]["exactitude"] == 0.75

    def test_empty_dataset_no_history(self, agent):
        result = agent.evaluate_llm_judge([])
        assert "error" in result
        assert agent.history == []

    def test_no_client_returns_error(self, agent):
        agent._openai_client = None
        result = agent.evaluate_llm_judge([{"question": "Q", "contexts": ["c"], "answer": "A"}])
        assert "error" in result


# ═══════════════════════════════════════════════════════════════
# 5. TESTS — FLAG_FOR_HUMAN_REVIEW
# ═══════════════════════════════════════════════════════════════

class TestFlagForHumanReview:

    def test_critical_threshold_flags_low_scores(self, agent):
        results = [
            {"id": "a", "faithfulness": 0.2},  # sous critical_threshold (0.5)
            {"id": "b", "faithfulness": 0.9},  # bon score, loin du seuil
        ]
        flagged = agent.flag_for_human_review(results)
        reasons = {f["id"]: f["reason"] for f in flagged}
        assert reasons.get("a") == "seuil_critique"

    def test_grey_zone_flags_scores_near_threshold(self, agent):
        # threshold=0.7, tolerance=0.10 => zone grise = [0.6, 0.8]
        results = [{"id": "a", "faithfulness": 0.72}]
        flagged = agent.flag_for_human_review(results)
        assert flagged[0]["reason"] == "zone_grise"

    def test_random_sample_included_on_good_scores(self, agent):
        results = [{"id": f"case_{i}", "faithfulness": 0.95} for i in range(20)]
        flagged = agent.flag_for_human_review(results)
        assert len(flagged) >= 1
        assert all(f["reason"] == "echantillon_aleatoire" for f in flagged)

    def test_missing_score_key_ignored(self, agent):
        results = [{"id": "a"}]  # pas de faithfulness
        flagged = agent.flag_for_human_review(results)
        assert flagged == []

    def test_empty_results(self, agent):
        assert agent.flag_for_human_review([]) == []


# ═══════════════════════════════════════════════════════════════
# 6. TESTS — GENERATE_HUMAN_REVIEW_SHEET & INTEGRATE_FEEDBACK
# ═══════════════════════════════════════════════════════════════

class TestHumanReviewSheet:

    def test_generate_sheet_structure(self, agent):
        flagged = [{"id": "case_0", "question": "Q?", "answer": "A.", "reason": "zone_grise"}]
        sheets = agent.generate_human_review_sheet(flagged)
        assert sheets[0]["id"] == "case_0"
        assert sheets[0]["decision"] is None
        assert "case_0" in agent.human_reviews

    def test_integrate_valid_decision(self, agent):
        flagged = [{"id": "case_0", "question": "Q?", "answer": "A.", "reason": "zone_grise"}]
        agent.generate_human_review_sheet(flagged)
        success = agent.integrate_human_feedback("case_0", "valide", "OK")
        assert success is True
        assert agent.human_reviews["case_0"]["decision"] == "valide"
        assert "reviewed_at" in agent.human_reviews["case_0"]

    def test_integrate_invalid_decision_rejected(self, agent):
        flagged = [{"id": "case_0", "question": "Q?", "answer": "A."}]
        agent.generate_human_review_sheet(flagged)
        success = agent.integrate_human_feedback("case_0", "peut-etre", "")
        assert success is False

    def test_integrate_unknown_case_id(self, agent):
        success = agent.integrate_human_feedback("inconnu", "valide")
        assert success is False


# ═══════════════════════════════════════════════════════════════
# 7. TESTS — COMPARE_VERSIONS & GET_TREND
# ═══════════════════════════════════════════════════════════════

class TestCompareVersionsAndTrend:

    def test_compare_versions_detects_regression(self, agent_with_history):
        diff = agent_with_history.compare_versions("v1", "v2")
        assert diff["faithfulness"]["trend"] == "régression"
        assert diff["answer_relevancy"]["trend"] == "régression"

    def test_compare_versions_missing_id(self, agent_with_history):
        diff = agent_with_history.compare_versions("v1", "v_inconnu")
        assert "error" in diff

    def test_get_trend_returns_all_points(self, agent_with_history):
        trend = agent_with_history.get_trend("faithfulness")
        assert len(trend) == 2
        assert trend[0][1] == 0.65
        assert trend[1][1] == 0.625

    def test_get_trend_unknown_metric_returns_empty(self, agent_with_history):
        trend = agent_with_history.get_trend("metrique_inexistante")
        assert trend == []


# ═══════════════════════════════════════════════════════════════
# 8. TESTS — EXPORTS
# ═══════════════════════════════════════════════════════════════

class TestExports:

    def test_export_json_valid(self, agent_with_history):
        content = agent_with_history.export(format="json")
        parsed = json.loads(content)
        assert "history" in parsed
        assert len(parsed["history"]) == 2

    def test_export_csv_has_expected_columns(self, agent_with_history):
        content = agent_with_history.export(format="csv")
        assert "faithfulness" in content
        assert "case_0" in content

    def test_export_markdown_contains_sections(self, agent_with_history):
        content = agent_with_history.export(format="markdown")
        assert "# Rapport d'évaluation RAG" in content
        assert "Comparaison des 2 dernières versions" in content

    def test_export_invalid_format_raises(self, agent_with_history):
        with pytest.raises(ValueError):
            agent_with_history.export(format="yaml")

    def test_export_writes_file(self, agent_with_history, tmp_path):
        filepath = tmp_path / "rapport.json"
        agent_with_history.export(format="json", filepath=str(filepath))
        assert filepath.exists()
        content = json.loads(filepath.read_text(encoding="utf-8"))
        assert "history" in content

    def test_export_csv_empty_history_returns_empty_string(self, agent):
        content = agent.export(format="csv")
        assert content == ""


# ═══════════════════════════════════════════════════════════════
# 9. TESTS — DASHBOARD
# ═══════════════════════════════════════════════════════════════

class TestDashboard:

    def test_dashboard_generated_file_exists(self, agent_with_history, tmp_path):
        output_path = tmp_path / "dashboard.html"
        result_path = agent_with_history.generate_dashboard(str(output_path))
        assert os.path.exists(result_path)
        content = output_path.read_text(encoding="utf-8")
        assert "Chart.js" in content or "chart.js" in content
        assert "radarChart" in content

    def test_dashboard_with_empty_history_does_not_crash(self, agent, tmp_path):
        output_path = tmp_path / "dashboard_empty.html"
        result_path = agent.generate_dashboard(str(output_path))
        assert os.path.exists(result_path)


# ═══════════════════════════════════════════════════════════════
# 10. TESTS — SUMMARY
# ═══════════════════════════════════════════════════════════════

class TestSummary:

    def test_summary_no_history(self, agent):
        result = agent.summary()
        assert "Aucune évaluation" in result

    def test_summary_with_history_includes_scores(self, agent_with_history):
        result = agent_with_history.summary()
        assert "faithfulness" in result
        assert "Cas signalés" in result

    def test_summary_detects_regression(self, agent_with_history):
        result = agent_with_history.summary()
        assert "Régression détectée" in result or "régression" in result.lower()

    def test_summary_recommends_on_low_scores(self, agent_with_history):
        result = agent_with_history.summary()
        assert "Recommandations" in result


# ═══════════════════════════════════════════════════════════════
# 11. TESTS — PROMETHEUS (optionnel)
# ═══════════════════════════════════════════════════════════════

class TestPrometheusExport:

    def test_export_skipped_if_unavailable(self, agent_with_history):
        with patch.object(rea, "PROMETHEUS_AVAILABLE", False):
            result = agent_with_history.export_to_prometheus()
            assert result is False

    def test_export_no_history_returns_false(self, agent):
        with patch.object(rea, "PROMETHEUS_AVAILABLE", True):
            result = agent.export_to_prometheus()
            assert result is False

    def test_export_success(self, agent_with_history):
        with patch.object(rea, "PROMETHEUS_AVAILABLE", True), \
             patch.object(rea, "push_to_gateway") as mock_push, \
             patch.object(rea, "Gauge") as mock_gauge, \
             patch.object(rea, "CollectorRegistry"):
            mock_gauge.return_value = MagicMock()
            result = agent_with_history.export_to_prometheus()
            assert result is True
            mock_push.assert_called_once()

    def test_export_handles_push_exception(self, agent_with_history):
        with patch.object(rea, "PROMETHEUS_AVAILABLE", True), \
             patch.object(rea, "push_to_gateway", side_effect=Exception("network error")), \
             patch.object(rea, "Gauge") as mock_gauge, \
             patch.object(rea, "CollectorRegistry"):
            mock_gauge.return_value = MagicMock()
            result = agent_with_history.export_to_prometheus()
            assert result is False


# ═══════════════════════════════════════════════════════════════
# 12. TEST D'INTÉGRATION — SCÉNARIO COMPLET
# ═══════════════════════════════════════════════════════════════

class TestIntegrationScenario:
    """Vérifie que l'enchaînement complet fonctionne sans erreur,
    du run RAGAS jusqu'à l'export, avec mocks pour les parties externes."""

    def test_full_workflow_runs_without_crash(self, agent, test_dataset, tmp_path):
        # Simule un run RAGAS déjà en historique (pour éviter de mocker
        # toute la chaîne Dataset/evaluate ici, déjà testée séparément)
        agent.history.append({
            "id": "v1",
            "timestamp": datetime.now().isoformat(),
            "type": "ragas",
            "scores": [
                {"id": "case_0", "question": "Q1", "answer": "A1", "faithfulness": 0.3},
                {"id": "case_1", "question": "Q2", "answer": "A2", "faithfulness": 0.9},
            ],
            "averages": {"faithfulness": 0.6},
        })

        flagged = agent.flag_for_human_review(agent.history[0]["scores"])
        sheets = agent.generate_human_review_sheet(flagged)
        if sheets:
            agent.integrate_human_feedback(sheets[0]["id"], "valide", "test ok")

        dashboard_path = agent.generate_dashboard(str(tmp_path / "dash.html"))
        json_export = agent.export(format="json", filepath=str(tmp_path / "r.json"))
        md_export = agent.export(format="markdown", filepath=str(tmp_path / "r.md"))
        summary_text = agent.summary()

        assert os.path.exists(dashboard_path)
        assert os.path.exists(tmp_path / "r.json")
        assert os.path.exists(tmp_path / "r.md")
        assert "faithfulness" in summary_text