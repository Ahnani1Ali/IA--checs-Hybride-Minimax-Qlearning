# ♟️ Chess AI 
> **Moteur d'échecs intelligent** combinant livre d'ouvertures, algorithme Minimax + Alpha-Bêta et Q-Learning par self-play.

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![python-chess](https://img.shields.io/badge/python--chess-1.10+-orange)](https://python-chess.readthedocs.io)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table des matières

- [Vue d'ensemble](#vue-densemble)
- [Architecture](#architecture)
- [Installation](#installation)
- [Démarrage rapide](#démarrage-rapide)
- [Composants](#composants)
  - [Règles d'échecs](#1-règles-déchecs)
  - [Livre d'ouvertures](#2-livre-douvertures)
  - [Minimax + Alpha-Bêta](#3-minimax--alpha-bêta)
  - [Q-Learning](#4-q-learning)
  - [Agent Hybride](#5-agent-hybride)
- [Notebook Jupyter](#notebook-jupyter)
- [Rapport LaTeX](#rapport-latex)
- [Résultats](#résultats)
- [Pistes d'amélioration](#pistes-damélioration)

---

## Vue d'ensemble

Ce projet implémente un moteur d'échecs de niveau M1 avec trois approches complémentaires :

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| Règles du jeu | `python-chess` | Mouvements légaux, FEN/PGN |
| Phase d'ouverture | Livre intégré + Polyglot | Séquences théoriques |
| Milieu de jeu | Minimax + α-β | Décision principale |
| Apprentissage | Q-Learning + Self-play | Amélioration continue |

---

## Architecture

```
chess_ai/
├── src/
│   ├── engine/
│   │   ├── board.py          # Wrapper python-chess (règles, FEN, PGN)
│   │   ├── evaluator.py      # Évaluation : matériel, PST, mobilité, centre
│   │   └── minimax.py        # Minimax + Alpha-Bêta + Quiescence Search
│   ├── opening/
│   │   └── opening_book.py   # Ouvertures intégrées + support Polyglot
│   ├── rl/
│   │   └── q_learning.py     # Q-Learning ε-greedy + self-play
│   ├── utils/
│   │   └── visualization.py  # SVG, courbes matplotlib, heatmaps
│   └── agent.py              # Agent hybride principal (pipeline de décision)
├── notebooks/
│   └── chess_ai_main.ipynb   # Notebook Jupyter complet et documenté
├── data/
│   └── q_table.pkl           # Q-table sauvegardée (générée à l'entraînement)
├── tests/
│   └── test_*.py             # Tests unitaires
├── latex_report/
│   └── rapport.tex           # Rapport LaTeX complet
├── requirements.txt
└── README.md
```

---

## Installation

### Prérequis

- Python 3.10+
- pip

### Étapes

```bash
# Cloner le dépôt
git clone https://github.com/[username]/chess-ai.git
cd chess-ai

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances principales

```
python-chess>=1.10.0
matplotlib>=3.7.0
numpy>=1.24.0
jupyter>=1.0.0
```

---

## Démarrage rapide

### Jouer une partie complète

```python
import chess
from src.agent import ChessAI

# Créer l'agent hybride (Blancs)
ai_white = ChessAI(
    mode='hybrid',        # 'minimax' | 'rl' | 'hybrid'
    minimax_depth=4,
    color=chess.WHITE,
)

# Adversaire Minimax (Noirs)
ai_black = ChessAI(
    mode='minimax',
    minimax_depth=3,
    color=chess.BLACK,
)

# Lancer la partie
result = ai_white.play_game(opponent=ai_black, verbose=True)
print(f"Résultat : {result['result']}")
print(result['pgn'])
```

### Minimax sur une position

```python
from src.engine.minimax import MinimaxAgent
import chess

agent = MinimaxAgent(depth=4, time_limit=5.0)
board = chess.Board()
board.push_uci('e2e4')
board.push_uci('e7e5')

best_move = agent.choose_move(board)
print(f"Meilleur coup : {best_move.uci()}")
```

### Entraînement Q-Learning

```python
from src.rl.q_learning import QLearningAgent

rl = QLearningAgent(alpha=0.3, gamma=0.95, epsilon=1.0)
rl.train(n_episodes=2000, verbose_every=100)
rl.save('data/q_table.pkl')
```

---

## Composants

### 1. Règles d'échecs

La classe `ChessBoard` encapsule `python-chess` :

```python
from src.engine.board import ChessBoard

cb = ChessBoard()
print(cb.get_legal_moves_uci())  # ['e2e4', 'd2d4', ...]
cb.push_uci('e2e4')
print(cb.is_check())   # False
print(cb.to_fen())     # 'rnbqkbnr/pppppppp/8/8/4P3/...'
```

**Fonctionnalités :**
- ✅ Mouvements légaux (toutes pièces)
- ✅ Roque (grand et petit), prise en passant, promotion
- ✅ Détection : échec, mat, pat, nulle par répétition
- ✅ Export FEN et PGN
- ✅ Hash Zobrist (table de transposition)

---

### 2. Livre d'ouvertures

```python
from src.opening.opening_book import OpeningBook

book = OpeningBook(random_weight=True)
move = book.get_move(board)      # coup de l'ouverture
name = book.get_opening_name(board)  # "Ruy Lopez"
```

**Ouvertures couvertes :**

| Ouverture | Ligne |
|-----------|-------|
| Ruy Lopez | 1.e4 e5 2.Nf3 Nc6 3.Bb5 |
| Défense sicilienne | 1.e4 c5 |
| Défense française | 1.e4 e6 |
| Gambit Dame | 1.d4 d5 2.c4 |
| Partie italienne | 1.e4 e5 2.Nf3 Nc6 3.Bc4 |
| English Opening | 1.c4 |

Support optionnel des fichiers **Polyglot** (`.bin`) pour un livre plus riche.

---

### 3. Minimax + Alpha-Bêta

```
Profondeur 4 :  ~12 000 nœuds · < 0.5s
Profondeur 5 :  ~100 000 nœuds · 5-15s
```

**Optimisations intégrées :**
- **Tri des coups** (MVV-LVA) → améliore l'élagage de ~3×
- **Quiescence Search** → évite l'effet d'horizon
- **Table de transposition** → évite les calculs redondants
- **Iterative Deepening** → meilleure gestion du temps

**Fonction d'évaluation :**
```
E(p) = Matériel + PST (Piece-Square Tables) + Mobilité + Contrôle centre + Sécurité roi
```

---

### 4. Q-Learning

**Formalisation MDP :**

| Composant | Définition |
|-----------|------------|
| État `s` | Hash FEN de la position |
| Action `a` | Coup UCI (`e2e4`) |
| Récompense `r` | +1 victoire, -1 défaite, 0 nulle, ±0.1 capture |

**Règle de Bellman :**
```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

**Self-play** : l'agent joue contre lui-même et apprend de ses erreurs.

---

### 5. Agent Hybride

Pipeline de décision :

```
Position
   │
   ▼ (si ≤ 20 demi-coups)
[Livre d'ouvertures] ──→ coup trouvé ? → jouer
   │ non
   ▼ (si Q-table non vide)
[Q-Learning] ──────────→ Q-value significative ? → jouer
   │ non
   ▼
[Minimax + α-β] ────────→ meilleur coup calculé → jouer
```

---

## Notebook Jupyter

Le notebook `notebooks/chess_ai_main.ipynb` couvre :

1. **Installation & imports**
2. **Démonstration des règles** (affichage SVG, coups spéciaux)
3. **Reconnaissance d'ouvertures** (test sur 6 ouvertures)
4. **Visualisation de l'évaluation** (courbe au fil d'une partie)
5. **Benchmark Minimax** (profondeurs 1-4, nœuds & temps)
6. **Entraînement Q-Learning** (self-play, courbes de convergence)
7. **Partie hybride** (avec log des sources de décision)
8. **Résultats & visualisations** (tournoi, heatmaps)

```bash
jupyter notebook notebooks/chess_ai_main.ipynb
```

---

## Rapport LaTeX

Compiler le rapport :

```bash
cd latex_report
pdflatex rapport.tex
pdflatex rapport.tex  # 2e passe pour la table des matières
```

**Contenu du rapport :**
- Introduction & état de l'art
- Architecture détaillée avec diagrammes TikZ
- Formalisation mathématique (Minimax, Bellman, évaluation)
- Analyse de complexité (tableaux, graphiques pgfplots)
- Résultats expérimentaux et benchmarks
- Discussion et pistes d'amélioration
- Bibliographie (Shannon 1950, AlphaZero 2018, ...)

---

## Résultats

### Benchmark Minimax

| Profondeur | Nœuds (α-β) | Temps |
|-----------|-------------|-------|
| 3 | ~1 500 | < 0.1s |
| 4 | ~12 000 | ~0.5s |
| 5 | ~100 000 | ~5-15s |

### Tournoi interne (50 parties)

| Blanc vs Noir | Victoires B | Nulles | Défaites B |
|---------------|-------------|--------|------------|
| Minimax-d3 vs d2 | 76% | 14% | 10% |
| Hybride vs d3 | 64% | 20% | 16% |

---

## Pistes d'amélioration

- [ ] **DQN** : approximation de la Q-fonction par réseau de neurones
- [ ] **MCTS** : Monte Carlo Tree Search (comme AlphaZero)
- [ ] **Livre Polyglot** : intégrer `baron30.bin` ou `komodo.bin`
- [ ] **Tablebases Syzygy** : fins de partie optimales (≤7 pièces)
- [ ] **Interface web** : Flask + `chessboard.js`
- [ ] **Multithreading** : parallélisation de la recherche Minimax

---

## Auteurs

- [Prénom NOM] — [email]
- [Prénom NOM] — [email]

**Encadrant :** [Nom de l'encadrant]  
**Université :** [Nom de l'université] — Master 1 IA & Data Science  
**Année :** 2024-2025

---

## Licence

MIT License — voir [LICENSE](LICENSE)

---

*"Chess is not about winning. It's about understanding." — (Adapté)*
