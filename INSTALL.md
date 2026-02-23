# Guide d'installation

## Prérequis

- Python **3.10** ou supérieur
- pip
- (Optionnel) LaTeX pour compiler le rapport

## Installation rapide

```bash
# 1. Cloner / extraire le projet
cd chess_ai_project

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate.bat     # Windows CMD
# venv\Scripts\Activate.ps1     # Windows PowerShell

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Vérifier l'installation
python -m pytest tests/ -v
```

## Démarrage en 3 commandes

```bash
# Entraîner l'IA (auto-play, 500 parties)
python scripts/train.py --episodes 500

# Jouer contre l'IA
python scripts/play.py --depth 4

# Ouvrir le notebook
jupyter notebook notebooks/chess_ai_main.ipynb
```

## Compiler le rapport PDF

```bash
# Avec pdflatex installé (TeX Live / MiKTeX)
make report
# ou manuellement :
cd docs/latex_report
pdflatex rapport.tex && pdflatex rapport.tex
```

## Dépendances clés

| Package | Version | Rôle |
|---------|---------|------|
| `python-chess` | ≥ 1.10 | Moteur d'échecs |
| `matplotlib` | ≥ 3.7 | Visualisations |
| `numpy` | ≥ 1.24 | Calcul numérique |
| `jupyter` | ≥ 1.0 | Notebook interactif |
| `pytest` | ≥ 7.0 | Tests unitaires |

## Structure des dossiers (voir README.md pour le détail)

```
chess_ai_project/
├── src/          → Code source Python
├── notebooks/    → Notebook Jupyter
├── scripts/      → Scripts CLI (train, play, benchmark)
├── tests/        → Tests unitaires
├── config/       → Fichier de configuration YAML
├── data/         → Données (Q-table, livres d'ouvertures)
├── docs/         → Rapport LaTeX + figures
└── Makefile      → Raccourcis de commandes
```
