# Modèles sauvegardés

Ce dossier contient les Q-tables et modèles entraînés.

## Fichiers générés automatiquement

| Fichier | Description |
|---------|-------------|
| `q_table.pkl` | Q-table principale (self-play, format pickle) |
| `q_table_backup_*.pkl` | Sauvegardes automatiques |
| `benchmark_results.json` | Résultats du dernier tournoi |

## Générer la Q-table

```bash
python scripts/train.py --episodes 2000 --save data/models/q_table.pkl
```

## Reprendre un entraînement

```bash
python scripts/train.py --episodes 1000 --resume data/models/q_table.pkl
```
