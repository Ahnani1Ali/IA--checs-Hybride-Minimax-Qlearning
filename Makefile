# Makefile — Raccourcis de commandes pour Chess AI

.PHONY: install test train play benchmark report clean

## Installation des dépendances
install:
	pip install -r requirements.txt

## Lancer les tests unitaires
test:
	python -m pytest tests/ -v --tb=short

## Lancer les tests avec couverture
test-cov:
	python -m pytest tests/ --cov=src --cov-report=term-missing

## Entraîner l'agent Q-Learning (500 épisodes rapides)
train:
	python scripts/train.py --episodes 500 --verbose-every 50

## Entraîner l'agent Q-Learning (2000 épisodes complets)
train-full:
	python scripts/train.py --episodes 2000 --verbose-every 100

## Jouer contre l'IA
play:
	python scripts/play.py --depth 4 --mode hybrid

## Benchmark de performance
benchmark:
	python scripts/benchmark.py --games 10 --depths 2 3 4

## Compiler le rapport LaTeX
report:
	cd docs/latex_report && \
	pdflatex rapport.tex && \
	pdflatex rapport.tex && \
	pdflatex rapport.tex

## Ouvrir le notebook Jupyter
notebook:
	jupyter notebook notebooks/chess_ai_main.ipynb

## Nettoyer les fichiers temporaires
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	cd docs/latex_report && rm -f *.aux *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz 2>/dev/null || true
	@echo "✅ Nettoyage terminé"
