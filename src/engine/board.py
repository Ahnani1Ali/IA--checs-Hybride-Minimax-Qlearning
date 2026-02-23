"""
board.py — Wrapper autour de python-chess
Gère les règles d'échecs, FEN, PGN, mouvements légaux.
"""

import chess
import chess.pgn
import io
from typing import Optional, List


class ChessBoard:
    """
    Abstraction de la logique du jeu d'échecs via python-chess.
    Gère : mouvements légaux, échec, mat, pat, roque, en passant.
    """

    def __init__(self, fen: Optional[str] = None):
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()

    # ------------------------------------------------------------------ #
    #  Informations de base                                                #
    # ------------------------------------------------------------------ #

    def get_legal_moves(self) -> List[chess.Move]:
        """Retourne la liste de tous les coups légaux."""
        return list(self.board.legal_moves)

    def get_legal_moves_uci(self) -> List[str]:
        """Retourne les coups légaux en notation UCI (ex: 'e2e4')."""
        return [move.uci() for move in self.board.legal_moves]

    def is_check(self) -> bool:
        return self.board.is_check()

    def is_checkmate(self) -> bool:
        return self.board.is_checkmate()

    def is_stalemate(self) -> bool:
        return self.board.is_stalemate()

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def get_result(self) -> Optional[str]:
        """Retourne '1-0', '0-1', '1/2-1/2' ou None si la partie continue."""
        result = self.board.result()
        return None if result == '*' else result

    def turn(self) -> chess.Color:
        """Retourne chess.WHITE ou chess.BLACK selon qui joue."""
        return self.board.turn

    # ------------------------------------------------------------------ #
    #  Actions                                                             #
    # ------------------------------------------------------------------ #

    def push_move(self, move: chess.Move) -> bool:
        """Joue un coup. Retourne True si légal, False sinon."""
        if move in self.board.legal_moves:
            self.board.push(move)
            return True
        return False

    def push_uci(self, uci: str) -> bool:
        """Joue un coup en notation UCI. Retourne True si légal."""
        try:
            move = chess.Move.from_uci(uci)
            return self.push_move(move)
        except ValueError:
            return False

    def undo_move(self):
        """Annule le dernier coup joué."""
        if self.board.move_stack:
            self.board.pop()

    # ------------------------------------------------------------------ #
    #  Sérialisation                                                       #
    # ------------------------------------------------------------------ #

    def to_fen(self) -> str:
        """Exporte la position en FEN."""
        return self.board.fen()

    def to_pgn(self) -> str:
        """Exporte la partie en PGN."""
        game = chess.pgn.Game.from_board(self.board)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        return game.accept(exporter)

    @classmethod
    def from_pgn(cls, pgn_str: str) -> "ChessBoard":
        """Crée un ChessBoard à partir d'une chaîne PGN."""
        game = chess.pgn.read_game(io.StringIO(pgn_str))
        board = cls()
        for move in game.mainline_moves():
            board.push_move(move)
        return board

    # ------------------------------------------------------------------ #
    #  Affichage                                                           #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return str(self.board)

    def unicode_board(self) -> str:
        """Retourne l'échiquier en unicode (pour Jupyter)."""
        return self.board.unicode(invert_color=True)

    def piece_at(self, square: int) -> Optional[chess.Piece]:
        """Retourne la pièce à la case donnée (0-63)."""
        return self.board.piece_at(square)
