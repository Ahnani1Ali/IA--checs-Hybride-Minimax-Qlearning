"""
agent.py — Agent principal combinant Ouvertures + Minimax + Q-Learning.

Pipeline de décision :
  1. Livre d'ouvertures → coup connu de la base
  2. Q-Learning (si entraîné) → coup appris par self-play
  3. Minimax + Alpha-Bêta → fallback algorithmique

L'agent peut fonctionner en mode :
  - 'minimax'    : Minimax uniquement (rapide, fiable)
  - 'rl'         : Q-Learning uniquement (expérimental)
  - 'hybrid'     : Ouvertures → RL → Minimax (recommandé)
"""

import chess
from typing import Optional

from .engine.board import ChessBoard
from .engine.minimax import MinimaxAgent
from .opening.opening_book import OpeningBook
from .rl.q_learning import QLearningAgent


class ChessAI:
    """
    Agent IA complet pour les échecs.

    Parameters
    ----------
    mode : str
        'minimax' | 'rl' | 'hybrid'
    minimax_depth : int
        Profondeur Minimax (3-5 recommandé).
    minimax_time_limit : float, optional
        Limite de temps en secondes pour Minimax.
    polyglot_path : str, optional
        Chemin vers un fichier Polyglot.
    q_table_path : str, optional
        Chemin vers une Q-table sauvegardée.
    color : chess.Color
        Couleur jouée par l'IA (WHITE ou BLACK).
    """

    def __init__(self,
                 mode: str = 'hybrid',
                 minimax_depth: int = 4,
                 minimax_time_limit: Optional[float] = 5.0,
                 polyglot_path: Optional[str] = None,
                 q_table_path: Optional[str] = None,
                 color: chess.Color = chess.WHITE):
        assert mode in ('minimax', 'rl', 'hybrid'), \
            "Mode invalide. Choisir parmi : 'minimax', 'rl', 'hybrid'"

        self.mode = mode
        self.color = color

        # Composants
        self.opening_book = OpeningBook(polyglot_path=polyglot_path)
        self.minimax = MinimaxAgent(depth=minimax_depth, time_limit=minimax_time_limit)
        self.rl_agent = QLearningAgent(q_table_path=q_table_path)

        self._decision_log: list = []

    # ------------------------------------------------------------------ #
    #  Sélection du coup                                                   #
    # ------------------------------------------------------------------ #

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Sélectionne le meilleur coup selon le mode de l'agent.
        Retourne None si aucun coup légal.
        """
        if not list(board.legal_moves):
            return None

        source = "?"
        move = None

        if self.mode in ('hybrid', 'minimax', 'rl'):
            # Étape 1 : Livre d'ouvertures (pour tous les modes)
            if self.mode != 'rl':
                move = self.opening_book.get_move(board)
                if move:
                    source = "opening_book"

        if move is None:
            if self.mode == 'minimax':
                move = self.minimax.choose_move(board)
                source = "minimax"

            elif self.mode == 'rl':
                move = self.rl_agent.choose_move(board)
                source = "q_learning"

            elif self.mode == 'hybrid':
                # Essai Q-Learning si Q-table non vide
                if self.rl_agent.q_table:
                    rl_move = self.rl_agent.choose_move(board)
                    # Utiliser RL seulement si la Q-value est significative
                    s_key = self.rl_agent._state_key(board)
                    if rl_move and self.rl_agent.q_table.get(s_key):
                        move = rl_move
                        source = "q_learning"

                # Fallback Minimax
                if move is None:
                    move = self.minimax.choose_move(board)
                    source = "minimax"

        opening_name = self.opening_book.get_opening_name(board)
        self._decision_log.append({
            'ply': board.fullmove_number,
            'fen': board.fen(),
            'move': move.uci() if move else None,
            'source': source,
            'opening': opening_name,
        })

        return move

    # ------------------------------------------------------------------ #
    #  Partie complète                                                     #
    # ------------------------------------------------------------------ #

    def play_game(self, opponent: Optional['ChessAI'] = None,
                  max_moves: int = 200, verbose: bool = True) -> dict:
        """
        Joue une partie complète.
        Si opponent=None, joue contre lui-même (self-play).

        Returns
        -------
        dict avec 'result', 'pgn', 'moves', 'log'
        """
        board = chess.Board()
        ai_white = self if self.color == chess.WHITE else opponent
        ai_black = self if self.color == chess.BLACK else opponent

        if opponent is None:
            ai_white = self
            ai_black = self

        move_history = []
        move_count = 0

        while not board.is_game_over() and move_count < max_moves:
            current_ai = ai_white if board.turn == chess.WHITE else ai_black
            move = current_ai.choose_move(board)

            if move is None:
                break

            if verbose:
                turn_str = "Blancs" if board.turn == chess.WHITE else "Noirs"
                src = self._decision_log[-1]['source'] if self._decision_log else '?'
                opening = self._decision_log[-1]['opening'] if self._decision_log else ''
                print(f"  {board.fullmove_number}. [{turn_str}] {move.uci()} "
                      f"(via {src}) — {opening}")

            board.push(move)
            move_history.append(move.uci())
            move_count += 1

        result = board.result() if board.is_game_over() else '*'
        if verbose:
            print(f"\n[Résultat] {result} après {move_count} demi-coups.")

        chess_board = ChessBoard.__new__(ChessBoard)
        chess_board.board = board

        return {
            'result': result,
            'pgn': chess_board.to_pgn(),
            'moves': move_history,
            'log': self._decision_log.copy(),
        }

    # ------------------------------------------------------------------ #
    #  Entraînement RL                                                     #
    # ------------------------------------------------------------------ #

    def train_rl(self, n_episodes: int = 1000, verbose_every: int = 100):
        """Lance l'entraînement du module Q-Learning par self-play."""
        self.rl_agent.train(n_episodes=n_episodes, verbose_every=verbose_every)

    def save_rl(self, path: str):
        """Sauvegarde la Q-table de l'agent RL."""
        self.rl_agent.save(path)

    # ------------------------------------------------------------------ #
    #  Infos                                                               #
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        return {
            'mode': self.mode,
            'color': 'Blancs' if self.color == chess.WHITE else 'Noirs',
            'minimax_depth': self.minimax.depth,
            'rl_stats': self.rl_agent.export_stats(),
        }

    def __repr__(self) -> str:
        color = 'Blancs' if self.color == chess.WHITE else 'Noirs'
        return f"ChessAI(mode={self.mode}, color={color}, depth={self.minimax.depth})"
