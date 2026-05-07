# ⚕️ MediAssist — Assistant Médicaments RAG

Système de Retrieval-Augmented Generation (RAG) pour répondre à des questions sur les médicaments courants, basé sur la Base de Données Publique des Médicaments (BDPM). Construit sans LangChain ni LlamaIndex — chaque brique est implémentée from scratch.

---

##  Structure du projet

```
.
├── app.py                  # Interface web Streamlit
├── rag.py                  # Interface en ligne de commande (RAG 2 agents)
├── indexer.py              # Script d'indexation (à lancer une fois)
├── find_links.py           # Utilitaire pour trouver les liens BDPM
├── requirements.txt        # Dépendances Python
├── .env                    # Clé API Groq (non commitée)
├── src/
│   ├── loader.py           # Chargement des données BDPM
│   ├── chunker.py          # Découpage en chunks avec overlap
│   ├── embedder.py         # Embeddings et gestion de l'index FAISS
│   └── agents.py           # RetrieverAgent et GeneratorAgent
└── data/
    └── index/              # Index FAISS + métadonnées (généré par indexer.py)
```

---

##  Installation

### 1. Cloner le projet et créer un environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Configurer la clé API Groq

Créez un compte gratuit sur [console.groq.com](https://console.groq.com), générez une clé API, puis créez un fichier `.env` à la racine :

```
GROQ_API_KEY=votre_clé_ici
```

### 3. Indexer les données (une seule fois)

```bash
python indexer.py
```

Cette commande charge les données médicaments depuis la BDPM, les découpe en chunks, crée les embeddings et sauvegarde l'index FAISS sur disque.

---

## Lancer l'application

### Interface en ligne de commande

```bash
python rag.py
```

Options disponibles :

```bash
python rag.py --no-bonus    # Désactive la reformulation de question
```

Commandes disponibles dans la CLI :
- `sources` — affiche les extraits utilisés pour la dernière réponse
- `quit` / `q` — quitte l'application

### Interface web (Streamlit)

```bash
streamlit run app.py
```

---

## Architecture

```
[Utilisateur]
     |
     v
[Agent 1 - Retriever]
  - Reformulation de la question (Bonus C)
  - Embedding de la question
  - Recherche vectorielle FAISS (top-k chunks)
  - Vérification du score de confiance (Bonus B)
     |
     v
[Agent 2 - Générateur]
  - Construction du prompt avec contexte
  - Appel API Groq (LLM)
  - Réponse avec sources et avertissement médical
     |
     v
[Réponse utilisateur]
```

### Modèles utilisés

| Composant | Modèle |
|-----------|--------|
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` (sentence-transformers) |
| LLM | `llama3-70b-8192` via Groq |
| Index vectoriel | FAISS `IndexFlatIP` (similarité cosinus) |

---

##  Médicaments couverts

Doliprane, Dafalgan, Efferalgan, Advil, Nurofen, Aspirin, Aspégic, Amoxicilline, Augmentin, Smecta, Imodium, Ventoline, Becotide, Oméprazole, Inexium, Metformine, Glucophage.

> Si une question dépasse ce périmètre, le système le signale explicitement sans inventer de réponse.

---

## Bonus implémentés

| Bonus | Description |
|-------|-------------|
| **A — Historique** | Le LLM tient compte des échanges précédents dans la conversation |
| **B — Score de confiance** | Si la similarité est trop faible, le système refuse de répondre plutôt que d'inventer |
| **C — Reformulation** | La question est reformulée par le LLM avant la recherche vectorielle pour améliorer la pertinence |

---

##  Dépendances

```
groq>=0.4.0
faiss-cpu>=1.7.4
sentence-transformers>=2.6.0
numpy>=1.24.0
python-dotenv>=1.0.0
requests>=2.31.0
tqdm>=4.66.0
streamlit>=1.30.0
BeautifulSoup4>=4.12.0
```

---

##  Avertissement

> Les informations fournies par cet assistant **ne remplacent pas l'avis d'un professionnel de santé**. En cas de doute, consultez votre médecin ou votre pharmacien.

---

---

## Réponses aux questions de réflexion préliminaires (Sujet B)

### R1 — Stratégie de chunking pour des notices longues et denses

Les notices médicaments sont des documents très longs (parfois 10 à 20 pages) avec une densité informationnelle élevée. Une taille de chunk de **800 à 1000 caractères** a été retenue, avec un **overlap de 100 caractères**.

Ce choix résulte d'un compromis :
- Un chunk trop petit (200-300 caractères) fragmente les phrases et perd le contexte (par exemple, une posologie peut s'étendre sur plusieurs phrases).
- Un chunk trop grand (2000+ caractères) noie le contenu pertinent dans du bruit et dépasse ce que le modèle d'embedding peut capturer efficacement.

L'overlap de 100 caractères permet de ne pas couper une information à cheval sur deux chunks et d'assurer une continuité sémantique entre morceaux adjacents.

### R2 — Exploitation de la structure des notices

Les notices médicaments sont structurées en sections bien identifiables : *Indications thérapeutiques*, *Posologie*, *Contre-indications*, *Effets indésirables*, *Interactions médicamenteuses*, etc.

Plutôt qu'un chunking purement mécanique basé sur le nombre de caractères, on effectue un **chunking structurel** : on repère les titres de sections (souvent en majuscules ou précédés de numéros comme `4.1`, `4.2`...) et on génère un chunk par section ou par sous-section. Si une section est trop longue, on la redécoupe avec l'overlap classique.

Avantages : chaque chunk est thématiquement cohérent, ce qui améliore la précision de la recherche vectorielle.

### R3 — Distinguer les types de chunks grâce aux métadonnées

Chaque chunk stocke dans ses métadonnées le champ `type_chunk`, qui indique la section dont il est issu : `posologie`, `effets_indesirables`, `contre_indications`, `interactions`, etc. Ainsi, lors de la recherche FAISS, on peut :

- **Filtrer** les résultats par type si la question est précise (ex : "effets secondaires" → privilégier `type_chunk = effets_indesirables`).
- **Afficher** le type dans l'interface pour que l'utilisateur sache d'où vient l'information.
- **Guider le LLM** en incluant le type dans le contexte fourni, permettant une réponse plus précise et mieux sourcée.

### R4 — Gérer les questions impliquant plusieurs médicaments

Une question comme *"Puis-je prendre du Doliprane et de l'ibuprofène en même temps ?"* nécessite des chunks des deux médicaments. Plusieurs stratégies ont été envisagées :

1. **Augmenter k** : récupérer plus de chunks (k=8 au lieu de 4) pour maximiser la probabilité d'avoir des informations sur les deux médicaments.
2. **Double requête** : détecter les noms de médicaments dans la question et lancer deux recherches distinctes, une par médicament, avant de fusionner les résultats.
3. **Reformulation ciblée** (Bonus C) : le LLM reformule la question en mentionnant explicitement les deux médicaments et le concept d'interaction, ce qui améliore la recherche vectorielle.

L'approche retenue combine la reformulation (Bonus C) et un k suffisamment large pour couvrir plusieurs médicaments.

### R5 — Formuler un prompt à la fois informatif et prudent

Le prompt système définit le LLM comme un **assistant d'information pharmaceutique**, non comme un médecin. Il intègre les contraintes suivantes :

- **Ne jamais inventer** : si l'information n'est pas dans le contexte fourni, dire explicitement qu'elle n'est pas disponible.
- **Citer systématiquement la source** : indiquer le médicament et la section (ex : *"D'après la notice officielle de l'ibuprofène, section Effets indésirables..."*).
- **Ajouter l'avertissement médical obligatoire** à chaque réponse : *"Ces informations ne remplacent pas l'avis d'un professionnel de santé."*
- **Ton pédagogique et accessible** : éviter le jargon médical brut, reformuler si nécessaire, sans pour autant dénaturer le contenu officiel.
- **Refus poli** : si la question dépasse le périmètre des médicaments indexés, le système le signale clairement plutôt que de répondre de manière approximative.
