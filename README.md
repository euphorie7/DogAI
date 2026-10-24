# 🐕 DogAI — Assistant RAG spécialisé dans les chiens

**DogAI** est un assistant conversationnel spécialisé dans les chiens, construit autour d'une architecture **RAG (Retrieval-Augmented Generation)** en Python, **sans LangChain**. Il recherche les passages pertinents dans une base documentaire, puis s'appuie sur ces passages pour générer une réponse à l'aide d'un LLM accessible via l'API Groq.

> **État du projet :** prototype fonctionnel en ligne de commande (CLI). Les embeddings et la recherche vectorielle fonctionnent localement ; la génération de réponses utilise une API distante.

## Fonctionnalités

- **Connaissances spécialisées** : questions sur les races de chiens, leurs caractéristiques et leur comportement.

- **Recherche sémantique** : récupération des `top_k` passages proches de la question, au-delà d'une simple recherche par mots-clés.

- **Embeddings locaux sur CPU** : modèle `BAAI/bge-small-en-v1.5` avec `sentence-transformers` (vecteurs de **384 dimensions**).

- **Stockage persistant** : ChromaDB conserve les chunks, les vecteurs et les métadonnées entre deux exécutions.

- **Génération contextualisée** : le modèle reçoit la question et les passages retrouvés, avec une instruction de citer les documents.

- **Réponses en français et en anglais**, selon la langue de la question.

- **Gestion du hors-sujet** : consignes invitant l'assistant à recentrer les échanges sur les chiens.

- **Indexation séparée de la recherche** : les embeddings du corpus ne sont pas recalculés à chaque question.

## Architecture

```mermaid

flowchart TD

&#x20; subgraph INDEX[Phase 1 — Indexation]

&#x20;   A[Knowledge Base : dogs.txt] --> B[Document Loader]

&#x20;   B --> C[Chunking : 500 caractères, overlap 100]

&#x20;   C --> D[BGE-small-en-v1.5]

&#x20;   D --> E[(ChromaDB : textes + embeddings)]

&#x20; end

&#x20; subgraph RAG[Phase 2 — Questions / Réponses]

&#x20;   F[Question utilisateur] --> G[Embedding de la question : BGE]

&#x20;   G --> H[Recherche par similarité]

&#x20;   E --> H

&#x20;   H --> I[Top-K passages]

&#x20;   I --> J[Prompt Builder]

&#x20;   J --> K[Groq API : GPT-OSS 20B]

&#x20;   K --> L[Réponse et références documentaires]

&#x20; end

```

### Stack technique

| Composant | Technologie | Rôle |

| --- | --- | --- |

| Langage | Python | Logique du pipeline et CLI |

| Knowledge Base | `data/dogs.txt` | Corpus de connaissances sur les chiens |

| Embeddings | `BAAI/bge-small-en-v1.5` | Vecteurs de 384 dimensions |

| Inférence des embeddings | `sentence-transformers` | Encodage en local sur CPU, par lots |

| Base vectorielle | ChromaDB | Stockage persistant et recherche sémantique |

| Retrieval | Recherche de similarité cosinus | Sélection des passages les plus pertinents |

| Génération | Groq API — `openai/gpt-oss-20b` | Génération de réponses |

| Configuration | `python-dotenv` | Chargement de la clé API |

## Dataset — attribution et licence

DogAI utilise des extraits du jeu de données **[Dog Breed N-Tuple Retrieval](https://huggingface.co/datasets/dorrito-dev/dog-breed-ntuple-retrieval)**, publié par **dorrito-dev** sur Hugging Face.

- **Source des textes :** articles de Wikipédia en anglais consacrés aux races de chiens ; crédits aux contributeurs des articles d'origine.
- **Licence indiquée pour le dataset :** [Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.fr).
- **Utilisation dans DogAI :** extraction des passages de la colonne `positive` du split `train`, suppression des doublons, tri puis concaténation dans `data/dogs.txt`, avec le séparateur `---DOCUMENT---`.
- **Modifications apportées :** sélection, déduplication et réorganisation du corpus pour l'indexation RAG ; les textes ne sont pas revendiqués comme des créations originales de DogAI.
- **En cas de redistribution des données :** conserver les crédits et le lien vers la licence, indiquer les modifications et appliquer les conditions CC BY-SA aux adaptations publiées. La licence du code source du projet est distincte de celle du corpus.

**Traçabilité :** le fichier `dogs.txt` regroupé ne conserve pas les URL de chaque article Wikipédia. Pour une attribution plus précise aux auteurs originaux, il est recommandé de conserver les liens et historiques des articles dont proviennent les passages.

## Structure du projet

Structure de référence issue du tutoriel (les chemins peuvent être adaptés à votre dépôt) :

```text

DogAI/

├── data/

│   └── dogs.txt

├── src/

│   ├── ingestion/

│   │   ├── loader.py

│   │   └── chunker.py

│   ├── embeddings/

│   │   └── embedder.py

│   ├── vectorstore/

│   │   └── store.py

│   ├── retrieval/

│   │   └── retriever.py

│   ├── generation/

│   │   ├── prompt.py

│   │   └── generator.py

│   ├── index.py

│   └── main.py

├── chroma_db/              # Généré localement, non versionné

├── .env                    # Clé API, non versionnée

├── .gitignore

├── requirements.txt

└── README.md

```

## Installation et exécution

### 1. Créer un environnement virtuel

```bash

python3 -m venv .venv

source .venv/bin/activate

```

Sous Windows (PowerShell) : `.venv\Scripts\Activate.ps1`.

### 2. Installer les dépendances

```bash

pip install -r requirements.txt

```

### 3. Configurer Groq

Créer un fichier `.env` à la racine du dépôt :

```dotenv

GROQ_API_KEY=votre_cle_groq

```

Créer une clé depuis la [console Groq](https://console.groq.com/keys). **Ne jamais publier `.env` ou une clé API dans Git.**

### 4. Préparer la base documentaire

Placer le corpus dans `data/dogs.txt`. Dans ce tutoriel, les passages proviennent d'un dataset Hugging Face consacré aux races de chiens : [dorrito-dev/dog-breed-ntuple-retrieval](https://huggingface.co/datasets/dorrito-dev/dog-breed-ntuple-retrieval).

Les passages peuvent être assemblés dans un seul fichier avec un séparateur explicite :

```text

Premier passage sur une race de chien.

---DOCUMENT---

Deuxième passage sur une autre race.

```

Le `Document Loader` peut ainsi séparer les documents avant le chunking. Le corpus est soumis à la licence **CC BY-SA 4.0** : voir la section « Dataset — attribution et licence » pour les crédits et conditions de redistribution.

### 5. Indexer la Knowledge Base

Depuis la racine du projet :

```bash

python src/index.py

```

Cette commande charge les textes, les découpe, génère les embeddings en local puis les stocke dans `./chroma_db`.

L'indexation est **distincte** du traitement des questions. Si une vérification `ensure_indexed()` est activée au démarrage, elle peut lancer l'indexation lorsque la collection est absente ou vide. Cette vérification **ne détecte pas** les mises à jour ultérieures du fichier `dogs.txt`.

### 6. Interroger DogAI

```bash

python src/main.py

```

Saisir une question puis `exit` pour quitter.

## Fonctionnement du RAG

**Indexation :** `dogs.txt` → chargement → chunks de **500 caractères** avec **100 caractères de chevauchement** → embeddings BGE → stockage ChromaDB.

**Interrogation :** question → embedding avec **le même modèle BGE** → recherche vectorielle dans la collection `dogs` → récupération des **3 meilleurs chunks** (`top_k=3`) → construction d'un prompt → appel Groq → réponse.

Le paramètre `batch_size=8` signifie que **8 chunks sont encodés par lot** sur le CPU. Il ne représente ni 8 caractères ni 8 dimensions. Chaque embedding produit par BGE-small-en-v1.5 comporte **384 valeurs**.

## Résultats observés

Les sorties suivantes proviennent des échanges de test de DogAI. Ce sont des **observations qualitatives**, et non des scores d'évaluation ni des avis vétérinaires validés.

### 1. Présentation et cadrage du domaine

**Question**

```text

Bonjour

```

**Réponse de DogAI**

> Bonjour ! Je suis DogAI, un assistant spécialisé uniquement dans les chiens. Je ne peux répondre qu’à des questions concernant les chiens. N’hésitez pas à me poser une question sur un chien ou une race de chien !

**Observation :** l'assistant annonce sa spécialisation. Un simple « Bonjour » ne suffit toutefois pas à démontrer un rejet fiable de toutes les questions hors sujet.

### 2. Réponse sur une race de chien avec citation

**Question**

```text

c'est quoi un berger malinois

```

**Réponse de DogAI**

> Un **berger malinois** (ou **Belgian Malinois**) est une variété de chien de berger belge. Il s’agit d’un chien de taille moyenne, très actif et intelligent, souvent utilisé comme chien de travail (police, armée, recherche et sauvetage). Selon l’étude mentionnée, il a une espérance de vie moyenne d’environ 12 ans. &#x20;

> **Référence : Document 1.**

**Observation :** réponse en français, accompagnée d'une référence au document récupéré. Les affirmations doivent être vérifiées dans la source originale.

### 3. Refus de compléter un contexte insuffisant — français

**Question**

```text

c'est quoi les symptomes de la rage

```

**Réponse de DogAI**

> Je n’ai pas assez d’informations dans le contexte fourni pour décrire les symptômes de la rage (ou du syndrome de rage). &#x20;

> [Document 1] ne donne que des indications génétiques et épidémiologiques, sans détailler les signes cliniques.

**Observation :** DogAI reconnaît une lacune documentaire. **Point de vigilance :** la rage virale (**rabies**) et le « rage syndrome » comportemental sont deux notions distinctes ; la réponse les rapproche à tort.

### 4. Refus de compléter un contexte insuffisant — anglais

**Question**

```text

what are the symptoms of rage

```

**Réponse de DogAI**

> I’m sorry, but I don’t have enough information in the provided context to list the symptoms of rage.

**Observation :** la langue de réponse suit celle de la question et l'assistant évite d'inventer des symptômes.

### 5. Question sur les problèmes de santé du berger malinois

**Question**

```text

quelle sont les maladies d'un berger malinois

```

**Réponse de DogAI**

> Les maladies les plus fréquemment associées au berger malinois sont :

>

> 1. **Dysplasie de la hanche**

> 2. **Dysplasie du coude**

> 3. **Épilepsie** (recommandée pour les lignes de travail)

> 4. **Syndrome de rage** (lié à une polymorphisme génétique du transporteur de dopamine)

>

> Ces recommandations proviennent de la Société Royale Saint‑Hubert et d’une étude britannique. &#x20;

> **Document 1**【1】.

**Observation :** cette réponse est reproduite comme résultat du système, **pas comme une liste médicalement validée de maladies fréquentes**. Certaines formulations sont confuses ; une vérification des sources et un contrôle de factualité sont nécessaires.

### 6. Recherche sans réponse suffisante

**Question**

```text

rabies ?

```

**Réponse de DogAI**

> I’m sorry, but I don’t have enough information on that topic.

**Observation :** le corpus ou le retrieval ne fournit pas de contexte suffisant pour répondre à cette question.

### 7. Recherche sur le « rage syndrome »

**Question**

```text

rage syndrome ?

```

**Réponse de DogAI**

> Rage syndrome is a behavioral condition that has been linked to the working lines of the Malinois. It is associated with a genetic polymorphism in the dopamine‑transporter gene, which can predispose dogs to this aggressive behavior. (Document 1)

**Observation :** le RAG récupère un passage lié au thème et le cite. **Le lien causal précis évoqué par la réponse n'est pas validé ici** ; cet exemple illustre la nécessité d'évaluer la fidélité à la source.

## Bilan qualitatif

| Critère | Constat pendant les essais |

| --- | --- |

| Recherche sémantique | Des passages sur le malinois sont récupérés |

| Réponses contextualisées | Réponses avec références documentaires observées |

| Français / anglais | Réponses dans les deux langues observées |

| Gestion de l'information manquante | Plusieurs refus explicites d'inventer |

| Pertinence et factualité | Encore à mesurer systématiquement |

| Désambiguïsation | Confusion détectée entre « rabies » et « rage syndrome » |

| Citations | Numéros fournis, mais pas encore de traçabilité fine (URL, titre, passage) |

> **Important :** aucune métrique quantitative (Recall@K, MRR, faithfulness, taux d'hallucination) n'a encore été calculée. Le bon fonctionnement des exemples ne prouve pas à lui seul la fiabilité du système.

## Améliorations prévues

1. **Évaluer le retrieval** avec un jeu de questions annotées : Recall@K, MRR, Precision@K.

2. **Évaluer la génération** : fidélité au contexte (**faithfulness**), pertinence et contrôle des citations.

3. **Améliorer le chunking** : respecter les paragraphes, titres et limites des documents.

4. **Gérer les ambiguïtés** : distinguer les maladies infectieuses des syndromes comportementaux.

5. **Fiabiliser l'indexation** : IDs stables, détection des changements et réindexation propre.

6. **Exposer une API FastAPI** pour séparer backend et interface.

7. **Créer un frontend React**, puis étudier le déploiement sur Vercel avec un backend et une base vectorielle hébergés séparément.

## Sécurité et limites

- Les documents récupérés sont envoyés à **Groq** au moment de la génération : ne pas utiliser de corpus confidentiel sans vérification des conditions de traitement.

- La clé API est conservée dans `.env`, exclu du versionnement.

- Les consignes du prompt améliorent le cadrage, mais **ne constituent pas une garantie** contre le hors-sujet ou les hallucinations.

- DogAI est un projet **pédagogique** ; ses réponses liées à la santé animale ne remplacent pas l'avis d'un vétérinaire.

---

**DogAI** — Un projet d'apprentissage du RAG, de l'indexation vectorielle et de la génération augmentée par récupération, développé étape par étape en Python.