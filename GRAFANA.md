# Dashboards Grafana (barres, jauges, séries temporelles)

Stack **Pushgateway → Prometheus → Grafana**, indépendante de Phoenix. Grafana
affiche les métriques de RagEval en **barres**, **jauges** et **courbes**.

```
RagEval (Python) --push--> Pushgateway --scrape--> Prometheus --datasource--> Grafana
```

## 1. Lancer la stack

```bash
cd monitoring
docker compose -f docker-compose.grafana.yml up -d
docker compose -f docker-compose.grafana.yml ps
```

- Grafana    : http://localhost:3001  (login **admin / admin**, il proposera de changer)
- Prometheus : http://localhost:9090
- Pushgateway: http://localhost:9091

Le dashboard **« RagEval — Qualité & Performance »** est déjà provisionné
(datasource Prometheus + panneaux). Rien à importer à la main.

## 2. Alimenter le dashboard

Test immédiat avec des données synthétiques :

```bash
pip install prometheus_client      # si pas déjà installé
python grafana_smoke.py            # pousse ~12 points sur 1 min
```

Ouvre http://localhost:3001 → dashboard **RagEval** : les barres, la jauge et les
courbes se remplissent.

### Avec de vrais scores

`RagEval.export_to_prometheus()` pousse déjà les moyennes RAGAS (mêmes noms
`rag_<métrique>`), vers le Pushgateway par défaut :

```python
run = agent.evaluate_ragas(mon_dataset, version_id="v1")
agent.export_to_prometheus(pushgateway_url="localhost:9091", job_name="rageval")
```

## 3. Ce que montre le dashboard

- **Barres** : les 4 scores RAGAS de la dernière éval (rouge < 0,6 < orange < 0,8 < vert).
- **Jauge** : la fidélité (faithfulness) seule, en un coup d'œil.
- **Séries temporelles** : évolution des scores et de la latence/tokens à chaque push.
- **Stat** : latence (ms) et tokens du dernier appel.

## 4. Limite honnête (et la suite)

Prometheus est fait pour des **séries temporelles agrégées** — parfait pour barres,
jauges et courbes. En revanche, le **nuage de points *par cas* latence ↔ qualité**
(croiser deux métriques cas par cas) n'est **pas** son terrain : ça se fait mieux
dans le dashboard HTML local (Chart.js) ou reste dans Phoenix trace par trace.
Ce croisement peut être ajouté au dashboard HTML sur demande.

## Gestion mémoire

Chaque service est plafonné (128–384 Mo). Lorsque ces services ne sont pas utilisés :

```bash
cd monitoring
docker compose -f docker-compose.grafana.yml stop   # libère la RAM, garde tout
docker compose -f docker-compose.grafana.yml start
```
