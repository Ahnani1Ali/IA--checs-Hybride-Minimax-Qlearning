"""
opening_book.py — Reconnaissance et gestion des ouvertures classiques.

Deux sources possibles :
  1. Livre d'ouvertures intégré (dictionnaire Python, format FEN → coups)
  2. Fichier Polyglot (.bin) via python-chess si disponible

Ouvertures couvertes :
  - Ruy Lopez (Lopez espagnol)
  - Sicilian Defense
  - French Defense
  - Queen's Gambit
  - Italian Game
  - King's Indian Defense
  - English Opening
"""

import chess
import chess.polyglot
import random
from pathlib import Path
from typing import Optional, Dict, List


# ------------------------------------------------------------------ #
#  Livre d'ouvertures intégré (FEN → liste de coups UCI)              #
# ------------------------------------------------------------------ #
#  Clé : FEN de la position (sans l'horloge)                          #
#  Valeur : liste de coups UCI avec poids (move, weight)              #

BUILTIN_BOOK: Dict[str, List[tuple]] = {
    # Position initiale
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -": [
        ("e2e4", 40), ("d2d4", 35), ("c2c4", 15), ("g1f3", 10),
    ],

    # 1. e4 — Réponses noires
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3": [
        ("e7e5", 35),  # Défense ouverte
        ("c7c5", 30),  # Sicilienne
        ("e7e6", 15),  # Française
        ("c7c6", 10),  # Caro-Kann
        ("d7d6",  5),  # Pirc
        ("g8f6",  5),  # Alekhine
    ],

    # 1. d4 — Réponses noires
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3": [
        ("d7d5", 40),  # Gambit Dame ou défense fermée
        ("g8f6", 30),  # Défense indienne
        ("f7f5", 10),  # Hollandaise
        ("e7e6", 20),  # Nimzo-indienne possible
    ],

    # Ruy Lopez — 1.e4 e5 2.Nf3 Nc6 3.Bb5
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq -": [
        ("a7a6", 40),  # Morphy defense (la plus populaire)
        ("f8c5", 20),  # Variante classique
        ("g8f6", 20),  # Berlin defense
        ("d7d6", 10),  # Steinitz defense
        ("g7g6", 10),  # Fianchetto
    ],

    # Sicilienne — 1.e4 c5 2.Nf3
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq -": [
        ("d7d6", 35),  # Najdorf / Dragon
        ("b8c6", 30),  # Rauzer / Richter
        ("e7e6", 25),  # Scheveningen
        ("g8f6", 10),  # Löwenthal
    ],

    # Défense française — 1.e4 e6 2.d4
    "rnbqkbnr/pppp1ppp/4p3/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq d3": [
        ("d7d5", 70),  # Coup principal
        ("c7c5", 20),  # Variante
        ("b8c6", 10),  # Variante
    ],

    # Gambit Dame — 1.d4 d5 2.c4
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3": [
        ("e7e6", 40),  # Gambit Dame refusé
        ("c7c6", 30),  # Slav defense
        ("d5c4", 20),  # Gambit Dame accepté
        ("g8f6", 10),  # Variante
    ],

    # Partie italienne — 1.e4 e5 2.Nf3 Nc6 3.Bc4
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq -": [
        ("f8c5", 50),  # Giuoco Piano
        ("g8f6", 30),  # Two Knights Defense
        ("f8e7", 10),  # Hungarian Defense
        ("h7h6", 10),  # Anti-pin
    ],

    # English Opening — 1.c4
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3": [
        ("e7e5", 40),  # Symétrique inversé
        ("c7c5", 30),  # Symétrique
        ("g8f6", 20),  # Variante
        ("e7e6", 10),  # Variante
    ],
}


class OpeningBook:
    """
    Gestionnaire du livre d'ouvertures.

    En début de partie : retourne un coup connu de l'ouverture.
    Sinon : retourne None (l'algorithme prend le relai).

    Parameters
    ----------
    polyglot_path : str, optional
        Chemin vers un fichier Polyglot .bin (ex: komodo.bin, baron30.bin).
    random_weight : bool
        Si True, choisit parmi les coups proposés avec probabilité pondérée.
        Si False, choisit toujours le coup avec le poids le plus élevé.
    max_opening_plies : int
        Nombre maximum de demi-coups de l'ouverture (0 = désactivé).
    """

    def __init__(self,
                 polyglot_path: Optional[str] = None,
                 random_weight: bool = True,
                 max_opening_plies: int = 20):
        self.polyglot_path = polyglot_path
        self.random_weight = random_weight
        self.max_opening_plies = max_opening_plies
        self._polyglot_reader = None

        if polyglot_path and Path(polyglot_path).exists():
            try:
                self._polyglot_reader = chess.polyglot.open_reader(polyglot_path)
                print(f"[OpeningBook] Fichier Polyglot chargé : {polyglot_path}")
            except Exception as e:
                print(f"[OpeningBook] Impossible de lire le fichier Polyglot : {e}")
        else:
            print("[OpeningBook] Mode intégré actif (livre Python).")

    # ------------------------------------------------------------------ #
    #  Interface publique                                                  #
    # ------------------------------------------------------------------ #

    def get_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Retourne un coup de l'ouverture pour la position donnée.
        Retourne None si la position n'est pas dans le livre,
        ou si le nombre de demi-coups dépasse max_opening_plies.
        """
        if self.max_opening_plies and board.fullmove_number * 2 > self.max_opening_plies:
            return None

        # Essai via Polyglot en priorité
        if self._polyglot_reader:
            move = self._get_polyglot_move(board)
            if move:
                return move

        # Fallback : livre intégré
        return self._get_builtin_move(board)

    def get_opening_name(self, board: chess.Board) -> Optional[str]:
        """Identifie le nom de l'ouverture en cours (approximatif)."""
        moves = [m.uci() for m in board.move_stack]

        if not moves:
            return "Position initiale"
        if moves[:2] == ["e2e4", "c7c5"]:
            return "Défense sicilienne"
        if moves[:2] == ["e2e4", "e7e6"]:
            return "Défense française"
        if moves[:3] == ["e2e4", "e7e5", "g1f3"]:
            if len(moves) >= 4 and moves[3] == "f1b5":
                return "Ruy Lopez (Lopez espagnol)"
            if len(moves) >= 4 and moves[3] == "f1c4":
                return "Partie italienne"
        if moves[:2] == ["d2d4", "d7d5"] and len(moves) >= 3 and moves[2] == "c2c4":
            return "Gambit Dame"
        if moves[:1] == ["c2c4"]:
            return "English Opening"
        return "Ouverture non identifiée"

    def close(self):
        """Ferme le lecteur Polyglot si ouvert."""
        if self._polyglot_reader:
            self._polyglot_reader.close()

    # ------------------------------------------------------------------ #
    #  Méthodes internes                                                   #
    # ------------------------------------------------------------------ #

    def _get_polyglot_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Lit un coup depuis le fichier Polyglot."""
        try:
            entries = list(self._polyglot_reader.find_all(board))
            if not entries:
                return None
            if self.random_weight:
                weights = [e.weight for e in entries]
                total = sum(weights)
                rand_val = random.uniform(0, total)
                cumul = 0
                for entry in entries:
                    cumul += entry.weight
                    if rand_val <= cumul:
                        return entry.move
            return max(entries, key=lambda e: e.weight).move
        except Exception:
            return None

    def _get_builtin_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Lit un coup depuis le livre intégré Python."""
        # Clé = FEN sans les compteurs d'horloge
        fen_key = " ".join(board.fen().split()[:4])
        candidates = BUILTIN_BOOK.get(fen_key)

        if not candidates:
            return None

        # Vérifier que les coups sont légaux
        legal_ucis = {m.uci() for m in board.legal_moves}
        valid = [(uci, w) for uci, w in candidates if uci in legal_ucis]
        if not valid:
            return None

        if self.random_weight:
            total = sum(w for _, w in valid)
            rand_val = random.uniform(0, total)
            cumul = 0
            for uci, w in valid:
                cumul += w
                if rand_val <= cumul:
                    return chess.Move.from_uci(uci)

        # Coup avec le poids maximum
        best_uci = max(valid, key=lambda x: x[1])[0]
        return chess.Move.from_uci(best_uci)
