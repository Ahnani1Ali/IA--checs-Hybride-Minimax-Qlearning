# Livres d'ouvertures (Polyglot)

Ce dossier est destiné à accueillir des fichiers de livres d'ouvertures au
format **Polyglot** (`.bin`), utilisés par le module `src/opening/opening_book.py`.

## Comment ajouter un livre

1. Télécharger un fichier `.bin` (voir sources ci-dessous)
2. Le placer dans ce dossier
3. Configurer le chemin dans `config/config.yaml` :

```yaml
opening_book:
  polyglot_path: data/opening_books/baron30.bin
```

## Sources recommandées (libres de droits)

| Fichier | Taille | Description |
|---------|--------|-------------|
| `baron30.bin` | ~1 MB | Livre standard, bonnes ouvertures |
| `komodo.bin`  | ~2 MB | Livre du moteur Komodo |
| `gm2001.bin`  | ~5 MB | Parties de grands maîtres |

Disponibles sur : https://www.chessprogramming.org/Polyglot

## Format

Un fichier Polyglot associe un **hash Zobrist** de position à une liste de
coups pondérés. python-chess le lit nativement via `chess.polyglot`.
