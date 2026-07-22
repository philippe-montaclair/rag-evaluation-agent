# Visualiser la qualité du RAG dans Phoenix (Arize)

Objectif : voir les scores d'évaluation (RAGAS niveau 1, juge niveau 2) dans
l'interface Phoenix, une trace par cas évalué. Le serveur tourne **en Docker**,
donc **rien de lourd à installer** côté Python — ça marche même en Python 3.13.

## 1. Lancer le serveur Phoenix (Docker)

```bash
cd ~/Desktop/Claude/Projects/agent_evaluation_rag
docker compose up -d          # démarre le conteneur en arrière-plan
```

- Interface web : http://localhost:6006
- Ingest OTLP (gRPC) : `localhost:4317` (utilisé automatiquement par l'export)
- Les traces sont persistées dans le volume `phoenix_data` (survivent aux redémarrages).

Arrêter : `docker compose down` (les données restent). Tout effacer : `docker compose down -v`.

## 2. Installer le client d'export (léger)

```bash
pip install -r requirements-phoenix.txt
```

SDK OpenTelemetry (émission des traces) **+ `arize-phoenix-client`** (rattache les
scores comme *mesures*). Tout est compatible Python 3.13.

## 3. Envoyer des scores

```python
from rag_evaluation_agent import RagEval
from phoenix_export import export_run_to_phoenix

agent = RagEval(pipeline=mon_pipeline, judge_model="gpt-4o-mini")

# Niveau 1 — RAGAS
run = agent.evaluate_ragas(mon_dataset, version_id="v1")
agent.export_to_phoenix(run, project_name="rag-locatif")

# Niveau 2 — juge (scores 0-1 normalisés)
run_j = agent.evaluate_llm_judge(mon_dataset, version_id="v1-judge")
agent.export_to_phoenix(run_j, project_name="rag-locatif")

# ...ou tout l'historique d'un coup :
agent.export_to_phoenix()          # run=None -> envoie tous les runs
```

Puis ouvre http://localhost:6006 : chaque cas apparaît comme une trace (question,
réponse, contextes) **et ses scores comme MESURES** (annotations `faithfulness`,
`answer_relevancy`, critères du juge…) — filtrables et triables dans l'UI.

`export_to_phoenix` renvoie `{"spans": n, "annotations": n}` : si `annotations` = 0
alors que `spans` > 0, c'est qu'`arize-phoenix-client` n'est pas installé (traces OK,
mais pas les mesures). Une attente de disponibilité de Phoenix est intégrée
(fini la course « StatusCode.UNAVAILABLE » au démarrage du conteneur).

## 4. Ce que ça couvre (et pas encore)

- ✅ **Qualité** : scores RAGAS + juge visualisables et filtrables dans Phoenix.
- ✅ **Performance** (latence + tokens) : via `phoenix_trace.py` (voir §5). Chaque appel
  du pipeline devient une trace avec sa durée (latence) et ses tokens si exposés.
  L'instrumentation *par étape* (retriever vs génération) via OpenInference reste
  une amélioration ultérieure.
- Grafana/Prometheus : `export_to_prometheus()` existe déjà pour les moyennes ;
  brancher un `docker-compose` Prometheus+Grafana viendra ensuite, derrière le
  même socle OpenTelemetry.

## 5. Tracer la performance du pipeline (latence + tokens)

`phoenix_trace.py` enrobe le pipeline sans le modifier :

```python
from rag_evaluation_agent import RagEval
from phoenix_trace import traced_pipeline

# 1. on enrobe le pipeline -> chaque appel est tracé (durée = latence)
pipe = traced_pipeline(mon_pipeline, project_name="rag-locatif")
agent = RagEval(pipeline=pipe, judge_model="gpt-4o-mini")

# 2. utilise l'agent normalement ; les traces partent vers Phoenix
resultat = agent.pipeline("Quel préavis pour un congé pour vente ?")

# 3. (optionnel) vider le buffer en fin de campagne
pipe.flush()
```

Dans l'UI (http://localhost:6006), chaque appel montre sa **latence** (durée du span)
et, si le pipeline renvoie les tokens, `llm.token_count.prompt/completion/total`.

Formes de tokens reconnues automatiquement :
- OpenAI : `result["usage"] = {"prompt_tokens", "completion_tokens", "total_tokens"}`
- simple : `result["token_count"] = <int total>`
Sinon, fournir un extracteur personnalisé : `traced_pipeline(pipe, token_extractor=ma_fonction)`.
Même sans tokens, la **latence est capturée gratuitement**.

## Dépannage

- **Rien dans l'UI** : vérifie que le conteneur tourne (`docker compose ps`) et que
  le port 4317 n'est pas pris. L'export logge « N cas envoyés » s'il a réussi.
- **`SDK OpenTelemetry absent`** : `pip install -r requirements-phoenix.txt`.
- **Attributs non reconnus comme « évaluations »** selon la version de Phoenix :
  ajuste les constantes en tête de `phoenix_export.py` (conventions OpenInference).
