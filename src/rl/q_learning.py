"""
q_learning.py — Agent Q-Learning pour les échecs (version simplifiée).

Architecture :
  - État  : hash Zobrist de la position (FEN tronqué)
  - Action: coup UCI (ex: 'e2e4')
  - Reward:
      +1.0  victoire
      -1.0  défaite
       0.0  nulle
      +0.1  prise d'une pièce adverse (par valeur)
      -0.1  prise d'une pièce propre

Fonctionnement :
  L'agent joue contre lui-même (self-play) et met à jour sa Q-table
  après chaque partie avec l'algorithme TD(0).

  Q(s, a) ← Q(s, a) + α [r + γ·max_a' Q(s', a') − Q(s, a)]

Note :
  Le Q-Learning pur sur échecs converge lentement (espace d'états ~10^44).
  Cette implémentation est pédagogique et montre les concepts.
  Pour de meilleures performances, utiliser le module mcts.py (Monte Carlo).
"""

import chess
import random
import json
import pickle
from pathlib import Path
from typing import Optional, Dict, Tuple
from collections import defaultdict


class QLearningAgent:
    """
    Agent Q-Learning avec exploration epsilon-greedy.

    Parameters
    ----------
    alpha   : taux d'apprentissage (0.1 à 0.5)
    gamma   : facteur d'actualisation (0.9 à 0.99)
    epsilon : taux d'exploration initial (1.0 = full random)
    epsilon_decay : facteur de réduction d'epsilon par épisode
    epsilon_min   : epsilon minimum (ex: 0.05 = 5% d'exploration résiduelle)
    """

    PIECE_CAPTURE_REWARDS = {
        chess.PAWN:   0.10,
        chess.KNIGHT: 0.32,
        chess.BISHOP: 0.33,
        chess.ROOK:   0.50,
        chess.QUEEN:  0.90,
        chess.KING:   0.00,  # Ne devrait pas arriver
    }

    def __init__(self,
                 alpha: float = 0.3,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.05,
                 q_table_path: Optional[str] = None):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table : dict[state_key][move_uci] = Q-value
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

        # Métriques d'entraînement
        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

        if q_table_path and Path(q_table_path).exists():
            self.load(q_table_path)

    # ------------------------------------------------------------------ #
    #  Interface publique                                                  #
    # ------------------------------------------------------------------ #

    def choose_move(self, board: chess.Board) -> Optional[chess.Move]:
        """Sélectionne un coup via epsilon-greedy."""
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None

        if random.random() < self.epsilon:
            return random.choice(legal_moves)

        return self._greedy_move(board, legal_moves)

    def update(self, state: chess.Board, move: chess.Move,
               reward: float, next_state: chess.Board, done: bool):
        """
        Mise à jour de la Q-table (TD update).

        Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
        """
        s_key = self._state_key(state)
        a_key = move.uci()
        q_old = self.q_table[s_key][a_key]

        if done:
            q_target = reward
        else:
            next_moves = list(next_state.legal_moves)
            if next_moves:
                next_q_values = [
                    self.q_table[self._state_key(next_state)][m.uci()]
                    for m in next_moves
                ]
                q_target = reward + self.gamma * max(next_q_values)
            else:
                q_target = reward

        self.q_table[s_key][a_key] = q_old + self.alpha * (q_target - q_old)

    def decay_epsilon(self):
        """Réduit epsilon après chaque épisode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    #  Self-Play Training                                                  #
    # ------------------------------------------------------------------ #

    def self_play_episode(self, max_moves: int = 200) -> Tuple[str, list]:
        """
        Joue une partie complète contre soi-même.
        Retourne (résultat, historique des transitions).

        Résultat possible : '1-0', '0-1', '1/2-1/2'
        """
        board = chess.Board()
        history: list = []  # (state_fen, move_uci, reward, next_fen, done)

        move_count = 0
        while not board.is_game_over() and move_count < max_moves:
            current_fen = board.fen()
            move = self.choose_move(board)

            if move is None:
                break

            # Calcul de la récompense intermédiaire (capture)
            intermediate_reward = self._capture_reward(board, move)
            is_white = (board.turn == chess.WHITE)

            board.push(move)
            move_count += 1
            done = board.is_game_over()

            # Récompense finale
            if done:
                result = board.result()
                if result == '1-0':
                    final_reward = 1.0 if is_white else -1.0
                elif result == '0-1':
                    final_reward = -1.0 if is_white else 1.0
                else:
                    final_reward = 0.0
                total_reward = intermediate_reward + final_reward
            else:
                total_reward = intermediate_reward

            history.append((current_fen, move.uci(), total_reward, board.fen(), done))

        # Rejouer l'historique pour mettre à jour la Q-table
        self._apply_updates(history)

        result = board.result() if board.is_game_over() else '*'
        self.episode_count += 1
        self.decay_epsilon()
        self._update_stats(result)

        return result, history

    def train(self, n_episodes: int = 1000, verbose_every: int = 100):
        """Lance n_episodes parties de self-play."""
        print(f"[Q-Learning] Début entraînement : {n_episodes} épisodes")
        for ep in range(n_episodes):
            result, _ = self.self_play_episode()
            if (ep + 1) % verbose_every == 0:
                total = self.wins + self.losses + self.draws
                wr = self.wins / total * 100 if total else 0
                print(f"  Ep {ep+1:5d} | ε={self.epsilon:.3f} | "
                      f"V:{self.wins} D:{self.draws} L:{self.losses} | "
                      f"WR:{wr:.1f}% | Q-states:{len(self.q_table)}")
        print("[Q-Learning] Entraînement terminé.")

    # ------------------------------------------------------------------ #
    #  Persistance                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        """Sauvegarde la Q-table (pickle)."""
        data = {
            'q_table': dict(self.q_table),
            'epsilon': self.epsilon,
            'episode_count': self.episode_count,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"[Q-Learning] Q-table sauvegardée → {path}")

    def load(self, path: str):
        """Charge une Q-table existante."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float), data['q_table'])
        self.epsilon = data.get('epsilon', self.epsilon)
        self.episode_count = data.get('episode_count', 0)
        self.wins = data.get('wins', 0)
        self.losses = data.get('losses', 0)
        self.draws = data.get('draws', 0)
        print(f"[Q-Learning] Q-table chargée depuis {path} ({len(self.q_table)} états)")

    def export_stats(self) -> dict:
        """Retourne les statistiques d'entraînement."""
        total = self.wins + self.losses + self.draws
        return {
            'episodes': self.episode_count,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.wins / total if total else 0,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
        }

    # ------------------------------------------------------------------ #
    #  Méthodes internes                                                   #
    # ------------------------------------------------------------------ #

    def _state_key(self, board: chess.Board) -> str:
        """Clé d'état = FEN tronqué (sans compteurs d'horloge)."""
        return " ".join(board.fen().split()[:4])

    def _greedy_move(self, board: chess.Board, legal_moves: list) -> chess.Move:
        """Choisit le coup avec la Q-value maximale."""
        s_key = self._state_key(board)
        best_move = None
        best_q = float('-inf')
        for move in legal_moves:
            q_val = self.q_table[s_key][move.uci()]
            if q_val > best_q:
                best_q = q_val
                best_move = move
        return best_move or random.choice(legal_moves)

    def _capture_reward(self, board: chess.Board, move: chess.Move) -> float:
        """Récompense intermédiaire pour une capture."""
        if not board.is_capture(move):
            return 0.0
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is None:
            # En passant
            return self.PIECE_CAPTURE_REWARDS[chess.PAWN]
        reward = self.PIECE_CAPTURE_REWARDS.get(captured_piece.piece_type, 0.0)
        return reward if board.turn == chess.WHITE else -reward

    def _apply_updates(self, history: list):
        """Applique les mises à jour TD sur l'historique de la partie."""
        for i, (s_fen, a_uci, reward, ns_fen, done) in enumerate(history):
            s_board = chess.Board(s_fen)
            ns_board = chess.Board(ns_fen)
            move = chess.Move.from_uci(a_uci)
            self.update(s_board, move, reward, ns_board, done)

    def _update_stats(self, result: str):
        if result == '1-0':
            self.wins += 1
        elif result == '0-1':
            self.losses += 1
        else:
            self.draws += 1
