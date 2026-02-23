"""
minimax.py — Agent Minimax avec élagage Alpha-Bêta.

Algorithme :
  1. Génère récursivement tous les coups jusqu'à la profondeur max.
  2. Évalue les feuilles avec Evaluator.
  3. Sélectionne le coup qui maximise le score (Blancs) ou le minimise (Noirs).
  4. L'Alpha-Bêta coupe les branches inutiles → ~10× plus rapide que Minimax pur.

Améliorations intégrées :
  - Tri des coups (captures et promotions en premier) pour améliorer l'élagage
  - Quiescence search pour éviter l'effet d'horizon
  - Cache transpositions (Zobrist hash via python-chess)
"""

import chess
import time
from typing import Optional, Tuple
from .evaluator import Evaluator

INF = float('inf')


class MinimaxAgent:
    """
    Moteur de jeu basé sur Minimax + Alpha-Bêta.

    Parameters
    ----------
    depth : int
        Profondeur de recherche (3-4 = niveau amateur fort, 5+ = semi-avancé).
    time_limit : float, optional
        Limite de temps en secondes (None = pas de limite).
    """

    def __init__(self, depth: int = 4, time_limit: Optional[float] = None):
        self.depth = depth
        self.time_limit = time_limit
        self.evaluator = Evaluator()
        self.nodes_visited = 0
        self._start_time: float = 0.0
        self._transposition_table: dict = {}

    # ------------------------------------------------------------------ #
    #  Interface publique                                                  #
    # ------------------------------------------------------------------ #

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Retourne le meilleur coup pour la position donnée.
        Retourne None si aucun coup légal n'existe.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        self.nodes_visited = 0
        self._start_time = time.time()
        self._transposition_table.clear()

        best_move, best_score = self._root_search(board)
        elapsed = time.time() - self._start_time
        print(f"[Minimax] Coup: {best_move.uci()}, Score: {best_score:.1f}, "
              f"Nœuds: {self.nodes_visited}, Temps: {elapsed:.2f}s")
        return best_move

    # ------------------------------------------------------------------ #
    #  Recherche principale                                                #
    # ------------------------------------------------------------------ #

    def _root_search(self, board: chess.Board) -> Tuple[Optional[chess.Move], float]:
        """Recherche à la racine — itère sur tous les coups légaux."""
        is_maximizing = (board.turn == chess.WHITE)
        best_move = None
        best_score = -INF if is_maximizing else INF

        # Iterative Deepening (basique)
        for depth in range(1, self.depth + 1):
            move_scores = []
            alpha, beta = -INF, INF

            for move in self._order_moves(board, list(board.legal_moves)):
                if self._time_exceeded():
                    break
                board.push(move)
                score = self._minimax(board, depth - 1, alpha, beta, not is_maximizing)
                board.pop()
                move_scores.append((move, score))

                if is_maximizing:
                    if score > best_score:
                        best_score = score
                        best_move = move
                    alpha = max(alpha, best_score)
                else:
                    if score < best_score:
                        best_score = score
                        best_move = move
                    beta = min(beta, best_score)

        return best_move, best_score

    def _minimax(self, board: chess.Board, depth: int,
                 alpha: float, beta: float, is_maximizing: bool) -> float:
        """
        Minimax récursif avec Alpha-Bêta pruning.

        Parameters
        ----------
        depth       : profondeur restante
        alpha       : meilleur score garanti pour le maximiseur
        beta        : meilleur score garanti pour le minimiseur
        is_maximizing : True si c'est au tour du maximiseur (Blancs)
        """
        self.nodes_visited += 1

        # Vérification terminale
        if board.is_game_over():
            return self.evaluator.evaluate(board)

        # Nœud feuille → Quiescence Search
        if depth == 0:
            return self._quiescence_search(board, alpha, beta, is_maximizing)

        # Transposition table lookup
        key = board.zobrist_hash()
        if key in self._transposition_table:
            cached = self._transposition_table[key]
            if cached['depth'] >= depth:
                return cached['score']

        moves = self._order_moves(board, list(board.legal_moves))

        if is_maximizing:
            max_eval = -INF
            for move in moves:
                if self._time_exceeded():
                    break
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Coupe Bêta (élagage)
            self._transposition_table[key] = {'depth': depth, 'score': max_eval}
            return max_eval
        else:
            min_eval = INF
            for move in moves:
                if self._time_exceeded():
                    break
                board.push(move)
                eval_score = self._minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Coupe Alpha (élagage)
            self._transposition_table[key] = {'depth': depth, 'score': min_eval}
            return min_eval

    # ------------------------------------------------------------------ #
    #  Quiescence Search                                                   #
    # ------------------------------------------------------------------ #

    def _quiescence_search(self, board: chess.Board, alpha: float,
                           beta: float, is_maximizing: bool, qdepth: int = 4) -> float:
        """
        Recherche de quiescence : continue uniquement les captures
        pour éviter l'effet d'horizon (évaluer des positions instables).
        """
        stand_pat = self.evaluator.evaluate(board)

        if qdepth == 0 or board.is_game_over():
            return stand_pat

        if is_maximizing:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            for move in self._order_moves(board, list(board.legal_moves), captures_only=True):
                board.push(move)
                score = self._quiescence_search(board, alpha, beta, False, qdepth - 1)
                board.pop()
                alpha = max(alpha, score)
                if alpha >= beta:
                    break
            return alpha
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
            for move in self._order_moves(board, list(board.legal_moves), captures_only=True):
                board.push(move)
                score = self._quiescence_search(board, alpha, beta, True, qdepth - 1)
                board.pop()
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return beta

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _order_moves(self, board: chess.Board, moves: list,
                     captures_only: bool = False) -> list:
        """
        Trie les coups pour améliorer l'élagage alpha-bêta :
          1. Promotions en dame
          2. Captures (MVV-LVA : Most Valuable Victim - Least Valuable Attacker)
          3. Autres coups
        """
        scored_moves = []
        for move in moves:
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                victim_val = victim.piece_type if victim else 0
                attacker_val = attacker.piece_type if attacker else 6
                score = 10 * victim_val - attacker_val + 1000
            elif move.promotion == chess.QUEEN:
                score = 900
            elif captures_only:
                continue
            else:
                score = 0
            scored_moves.append((score, move))

        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [m for _, m in scored_moves]

    def _time_exceeded(self) -> bool:
        if self.time_limit is None:
            return False
        return (time.time() - self._start_time) >= self.time_limit
