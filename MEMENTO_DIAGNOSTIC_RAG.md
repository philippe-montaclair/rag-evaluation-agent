# Mémento diagnostic RAG — interpréter un score anormal et savoir quoi corriger

Référence rapide pour l'agent d'évaluation (RagEval). Pour chaque signal faible :
**cause probable → action immédiate → piège de surapprentissage à éviter.**

> Règle d'or transversale : **un score bas n'est pas une consigne d'optimisation, c'est une
> hypothèse à recouper.** Chaque métrique a son propre bruit (modèle juge, embeddings, taille du
> jeu). On agit après recoupement, pas au premier chiffre.

---

## 0. Lire les scores : la décomposition « triade RAG »

Toute panne RAG se range dans **trois cases**. Localiser la case AVANT d'agir :

| Case | Question posée | Métriques qui la mesurent |
|------|----------------|---------------------------|
| **A. Récupération** | Ai-je ramené les bons extraits ? | `context_precision`, `context_recall`, TruLens *context relevance* |
| **B. Ancrage (génération)** | La réponse colle-t-elle aux extraits ? | `faithfulness`, TruLens *groundedness*, juge `absence_hallucination` |
| **C. Pertinence (génération)** | La réponse répond-elle à la question ? | `answer_relevancy`, juge `pertinence_reponse` |
| **D. Bout-en-bout** | La réponse est-elle juste vs vérité ? | `answer_correctness` (exige `ground_truth`) |

**Le diagnostic est dans le croisement A×B×C, pas dans un score isolé.** (cf. §2)

---

## 1. Table métrique par métrique

### `faithfulness` faible (ancrage) — *ex. observé : 0.61*
- **Symptôme** : la réponse contient des affirmations non traçables aux extraits (extrapolation, chiffres/délais ajoutés).
- **Action immédiate** :
  1. Durcir le **prompt de synthèse** : « uniquement à partir des extraits, n'ajoute aucun chiffre/délai/condition absent ». *(fait le 22/07 → juge fidélité 0.75→0.80)*
  2. `temperature = 0` sur la synthèse.
  3. Raccourcir la réponse (1-3 phrases) → moins de claims non supportés.
- **AVANT de blâmer la génération** : vérifier `context_recall`. Recall bas ⇒ l'info manque dans le contexte ⇒ le modèle « bouche les trous » en inventant ⇒ **corriger la récupération d'abord** (§ context_recall).
- **Piège surapprentissage** : sur-contraindre jusqu'au copier-coller verbatim gonfle faithfulness mais tue lisibilité et pertinence (Goodhart). Ne pas ajuster le prompt sur les mêmes questions qu'on mesure.

### `answer_relevancy` faible (pertinence) — *ex. observé : 0.47*
- **Symptôme** : réponse hors-sujet, verbeuse, avec préambule, ou partielle.
- **Comment RAGAS la calcule** : il régénère des questions à partir de la réponse et mesure leur proximité (embeddings) avec la question d'origine. Préambule/reformulation → dilution → score bas.
- **Action immédiate** :
  1. Prompt : **réponse directe dès le premier mot**, zéro introduction. *(fait le 22/07 → juge pertinence 0.90→0.95)*
  2. Vérifier le **modèle d'embeddings** de la mesure (`nomic-embed-text` sous-mesure parfois) → tester `mxbai-embed-large` ou `bge-m3`.
  3. Si réponse incomplète (pas hors-sujet) → problème de `context_recall`, pas de prompt.
- **Piège surapprentissage** : bourrer la réponse des mots de la question fait monter la métrique sans améliorer la vraie qualité.

### `context_precision` faible (récupération) — *ex. observé : 0.93 (OK)*
- **Symptôme** : des extraits hors-sujet remontent haut dans le top-k (bruit).
- **Action immédiate** :
  1. Activer/renforcer le **reranker** (CrossEncoder — déjà actif ici).
  2. **Baisser `k`** (moins de bruit).
  3. Meilleur **chunking** (chunks trop gros = diluent la pertinence).
  4. **Filtres métadonnées** / recherche hybride (BM25 + dense).
- **Piège surapprentissage** : réduire `k` trop fort casse le `recall`. Optimiser precision et recall **ensemble**.

### `context_recall` faible (récupération)
- **Symptôme** : l'extrait qui contient LA réponse n'est pas récupéré (exige `ground_truth`).
- **Action immédiate** :
  1. **Augmenter `k`** (ou `pool` avant rerank).
  2. Meilleur **chunking** : chunks plus petits + chevauchement.
  3. **Reformulation / expansion de requête** (HyDE, multi-query) — c'est ICI que le *query prompting* sert vraiment.
  4. Meilleur **modèle d'embeddings** ou **recherche hybride**.
  5. Vérifier que le document est **réellement indexé** (panne bête n°1).
- **Piège surapprentissage** : monter `k` sans limite noie la génération de distracteurs → `precision` ET `faithfulness` baissent.

### `answer_correctness` faible (bout-en-bout)
- **Symptôme** : réponse fausse vs `ground_truth`. **Métrique composite** (factuel + sémantique).
- **Action immédiate** : **décomposer** — regarder d'abord `faithfulness` (B) et `context_recall` (A). Vérifier aussi la **qualité du `ground_truth`** lui-même (une vérité erronée fausse tout).
- **Piège surapprentissage** : la part sémantique récompense la ressemblance de formulation ; ne pas caler les réponses sur le style du `ground_truth`.

### Critères juge LLM (`fidelite`, `pertinence`, `exactitude`, `clarte`…)
- **Symptôme** : scores instables ou en désaccord avec RAGAS.
- **Action immédiate** :
  1. **Un désaccord juge↔RAGAS est un signal, pas une erreur** : il révèle qu'une des deux mesures sous/sur-évalue. Trancher **à la main** sur 2-3 cas.
  2. Juge sur petit modèle local (7-8B) = bruité → `double_pass=True`, ou juge plus fort, ou échelle normalisée 0-1 (déjà en place).
- **Piège surapprentissage** : optimiser pour plaire au juge (surtout s'il est faible) ≠ améliorer la qualité réelle.

### Performance (latence / tokens — Phoenix)
- **Symptôme** : latence élevée, coût tokens.
- **Action immédiate** : modèle plus petit ; réduire le contexte (k, taille chunks) ; prompt caching ; streaming ; couper le « thinking » (`reasoning=False`).
- **Piège surapprentissage** : sacrifier la qualité pour la vitesse sans le mesurer conjointement (garder qualité + perf sur les mêmes traces).

---

## 2. Recoupements : le vrai diagnostic (A × B × C)

| Precision/Recall (A) | Faithfulness (B) | Relevancy (C) | Diagnostic | Action prioritaire |
|:---:|:---:|:---:|---|---|
| ✅ haut | ✅ haut | ✅ haut | RAG sain | Rien (ou perf) |
| ✅ haut | ❌ bas | ✅/❌ | **Génération invente** malgré bon contexte | **Prompt de synthèse** (grounding) |
| ✅ haut | ✅ haut | ❌ bas | Réponse ancrée mais **à côté** | **Prompt** (directivité) + embeddings de mesure |
| ❌ recall bas | ❌ bas | ❌ bas | **Contexte manquant** → le modèle comble en inventant | **Récupération** (k, chunking, reformulation) — PAS le prompt |
| ❌ precision bas | ✅ haut | ❌ bas | Trop de bruit récupéré, réponse noyée | **Reranker / baisser k / chunking** |
| ✅ haut | ✅ haut | ✅ haut MAIS `correctness` bas | Contexte OK, réponse plausible mais **fausse**, ou `ground_truth` douteux | Vérifier `ground_truth`, puis exactitude factuelle |

> **Cas de CE projet (22/07)** : precision 0.93 / recall 0.93 (A ✅), mais faithfulness 0.61 & relevancy 0.47 (B/C ❌) → ligne 2/3 → **c'est la génération, pas la récupération** → reranking/query prompting seraient inutiles ici. D'où le durcissement du prompt de synthèse.

---

## 3. Dysfonctions techniques (pipeline & mesure) — vues dans ce projet

| Panne | Signature | Correctif immédiat | Statut |
|---|---|---|---|
| **Tempête de timeouts RAGAS** | `TimeoutError` sur presque tous les Jobs | `RunConfig(max_workers=1, timeout=300)` — Ollama traite en série | ✅ corrigé |
| **Crash agrégation** | `Could not convert string '…' to numeric` | Ne moyenner que les colonnes **numériques** (schéma RAGAS varie : `user_input`/`response`…) | ✅ corrigé |
| **JSON cassé par modèle « thinking »** | `<think>` pollue la sortie, parse KO | `reasoning=False` (ChatOllama) / `format="json"` / modèle *instruct* (mistral) | ✅ corrigé |
| **Sortie `null` ponctuelle** | `OutputParserException … input_value=None` | Toléré (NaN écarté) ; sinon retry ou modèle plus strict | ⏳ étape 4 |
| **Métriques toutes NaN** | moyennes vides | Vérifier modèle juge/embeddings dispo ; garder NaN hors moyenne | ✅ géré |
| **Mélange d'échelles** | juge 1-5 vs seuils 0-1 | Normaliser (`_normalize_judge_score`), garder `_raw_1_5` | ✅ en place |
| **`ragas non installé`** | run dans `base` conda, pas le `venv` | `source venv/bin/activate` avant de lancer | ⚠️ vécu 22/07 |
| **Index absent/périmé** | recall à plancher, `chercher_loi` vide | Reconstruire l'index (`build_index.py`), vérifier la collection | à surveiller |

---

## 4. Anti-surapprentissage — les 6 garde-fous

1. **Jamais tuner sur le jeu qu'on mesure.** Séparer *dev* (pour itérer le prompt) et *test* (pour juger). Fuite de données = illusion de progrès.
2. **Petit échantillon = forte variance.** 5 questions ne prouvent rien ; confirmer sur le jeu complet (41) avant de conclure. Ne pas courir après les décimales.
3. **Les métriques sont des proxys bruités.** Un désaccord juge↔RAGAS se tranche **à la main** sur quelques cas, pas en croyant le chiffre.
4. **Goodhart** : « quand une mesure devient une cible, elle cesse d'être une bonne mesure ». Optimiser faithfulness au point de copier-coller détruit la lisibilité.
5. **Reproductibilité** : `temperature=0` et seed fixe pour le flagging, sinon on « améliore » du bruit.
6. **Une variable à la fois.** Changer prompt + k + modèle ensemble empêche de savoir ce qui a agi.

---

## 5. Arbre de décision express

```
Score faible ?
├─ context_recall bas ........... RÉCUPÉRATION : k↑, chunking, reformulation, vérifier l'index
├─ context_precision bas ........ RÉCUPÉRATION : reranker, k↓, chunking, filtres/ hybride
├─ faithfulness bas (recall OK) . GÉNÉRATION : prompt grounding, temp=0, réponse courte
├─ answer_relevancy bas ......... GÉNÉRATION : prompt directif ; vérifier embeddings de mesure
├─ answer_correctness bas ....... DÉCOMPOSER : faithfulness + recall ; vérifier ground_truth
├─ juge ↔ RAGAS divergent ....... MESURE : trancher à la main ; double_pass / juge plus fort
└─ latence/tokens hauts ......... PERF : modèle↓, contexte↓, caching, reasoning=False
```

*Dernière mise à jour : 22 juillet 2026. Complète le README et session_handoff.md.*
