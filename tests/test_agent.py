"""
tests/test_agent.py — Tests d'intégration pour l'agent Chess AI.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
import pytest
from src.agent import ChessAI


class TestChessAIAgent:

    def test_minimax_mode(self):
        ai = ChessAI(mode='minimax', minimax_depth=2, color=chess.WHITE)
        board = chess.Board()
        move = ai.choose_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_rl_mode(self):
        ai = ChessAI(mode='rl', color=chess.WHITE)
        board = chess.Board()
        move = ai.choose_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_hybrid_mode(self):
        ai = ChessAI(mode='hybrid', minimax_depth=2, color=chess.WHITE)
        board = chess.Board()
        move = ai.choose_move(board)
        assert move is not None
        assert move in board.legal_moves

    def test_play_short_game(self):
        white = ChessAI(mode='minimax', minimax_depth=2, color=chess.WHITE)
        black = ChessAI(mode='minimax', minimax_depth=2, color=chess.BLACK)
        result = white.play_game(opponent=black, max_moves=30, verbose=False)
        assert 'result' in result
        assert 'moves' in result
        assert 'pgn' in result
        assert len(result['moves']) > 0

    def test_game_result_valid(self):
        white = ChessAI(mode='minimax', minimax_depth=2, color=chess.WHITE)
        black = ChessAI(mode='minimax', minimax_depth=2, color=chess.BLACK)
        result = white.play_game(opponent=black, max_moves=50, verbose=False)
        assert result['result'] in {'1-0', '0-1', '1/2-1/2', '*'}

    def test_decision_log_populated(self):
        ai = ChessAI(mode='hybrid', minimax_depth=2, color=chess.WHITE)
        board = chess.Board()
        for _ in range(5):
            move = ai.choose_move(board)
            if move:
                board.push(move)
        assert len(ai._decision_log) >= 1
        assert 'source' in ai._decision_log[0]
        assert 'move' in ai._decision_log[0]

    def test_get_stats(self):
        ai = ChessAI(mode='hybrid', minimax_depth=3)
        stats = ai.get_stats()
        assert stats['mode'] == 'hybrid'
        assert stats['minimax_depth'] == 3
        assert 'rl_stats' in stats

    def test_invalid_mode_raises(self):
        with pytest.raises(AssertionError):
            ChessAI(mode='invalid_mode')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
