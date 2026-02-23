"""
evaluator.py — Fonction d'évaluation statique de la position.

Critères pris en compte :
  1. Matériel (valeur des pièces)
  2. Contrôle du centre
  3. Mobilité (nombre de coups légaux)
  4. Sécurité du roi (pénalités en cas d'échec)
  5. Tables de valeurs positionnelles (Piece-Square Tables)
"""

import chess
from typing import Dict

# ------------------------------------------------------------------ #
#  Valeurs des pièces (en centipawns)                                 #
# ------------------------------------------------------------------ #
PIECE_VALUES: Dict[chess.PieceType, int] = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

# ------------------------------------------------------------------ #
#  Piece-Square Tables (du point de vue des blancs)                   #
#  Basées sur les tables de Tomasz Michniewski (domaine public)       #
# ------------------------------------------------------------------ #

PAWN_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

KNIGHT_TABLE = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]

BISHOP_TABLE = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]

ROOK_TABLE = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

QUEEN_TABLE = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]

KING_MIDGAME_TABLE = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

PST = {
    chess.PAWN:   PAWN_TABLE,
    chess.KNIGHT: KNIGHT_TABLE,
    chess.BISHOP: BISHOP_TABLE,
    chess.ROOK:   ROOK_TABLE,
    chess.QUEEN:  QUEEN_TABLE,
    chess.KING:   KING_MIDGAME_TABLE,
}


class Evaluator:
    """
    Évaluateur heuristique de position.
    Score positif = avantage aux blancs.
    Score négatif = avantage aux noirs.
    """

    CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}
    EXTENDED_CENTER = {chess.C3, chess.C4, chess.C5, chess.C6,
                       chess.D3, chess.D6, chess.E3, chess.E6,
                       chess.F3, chess.F4, chess.F5, chess.F6}

    def evaluate(self, board: chess.Board) -> float:
        """
        Évalue la position courante.
        Retourne un score en centipawns (float).
        """
        if board.is_checkmate():
            # La couleur qui vient de jouer a mis en échec et mat
            return -20000 if board.turn == chess.WHITE else 20000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0.0
        score += self._material_score(board)
        score += self._positional_score(board)
        score += self._mobility_score(board)
        score += self._center_control_score(board)
        score += self._king_safety_score(board)
        return score

    # ------------------------------------------------------------------ #
    #  Composantes de l'évaluation                                        #
    # ------------------------------------------------------------------ #

    def _material_score(self, board: chess.Board) -> float:
        score = 0
        for piece_type, value in PIECE_VALUES.items():
            score += value * len(board.pieces(piece_type, chess.WHITE))
            score -= value * len(board.pieces(piece_type, chess.BLACK))
        return score

    def _positional_score(self, board: chess.Board) -> float:
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            table = PST.get(piece.piece_type, [0] * 64)
            if piece.color == chess.WHITE:
                # Miroir vertical pour les blancs (a1 = index 0)
                idx = chess.square_mirror(square)
                score += table[idx]
            else:
                score -= table[square]
        return score

    def _mobility_score(self, board: chess.Board) -> float:
        """Bonus pour le nombre de coups légaux disponibles."""
        white_mobility = 0
        black_mobility = 0
        # Coups légaux du joueur actuel
        current_mobility = board.legal_moves.count()
        if board.turn == chess.WHITE:
            white_mobility = current_mobility
            board.push(chess.Move.null())
            black_mobility = board.legal_moves.count()
            board.pop()
        else:
            black_mobility = current_mobility
            board.push(chess.Move.null())
            white_mobility = board.legal_moves.count()
            board.pop()
        return 0.1 * (white_mobility - black_mobility)

    def _center_control_score(self, board: chess.Board) -> float:
        """Bonus pour le contrôle du centre."""
        score = 0
        for sq in self.CENTER_SQUARES:
            score += 0.3 * len(board.attackers(chess.WHITE, sq))
            score -= 0.3 * len(board.attackers(chess.BLACK, sq))
        for sq in self.EXTENDED_CENTER:
            score += 0.1 * len(board.attackers(chess.WHITE, sq))
            score -= 0.1 * len(board.attackers(chess.BLACK, sq))
        return score

    def _king_safety_score(self, board: chess.Board) -> float:
        """Pénalité si le roi est en danger."""
        score = 0
        if board.is_check():
            # Pénalité légère pour la position d'échec en cours
            penalty = -50 if board.turn == chess.WHITE else 50
            score += penalty
        return score
