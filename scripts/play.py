#!/usr/bin/env python3
"""
scripts/play.py — Jouer contre l'IA en ligne de commande.

Usage :
    python scripts/play.py                          # Vous = Blancs, IA = Noirs
    python scripts/play.py --color black            # Vous = Noirs, IA = Blancs
    python scripts/play.py --depth 4                # Profondeur Minimax
    python scripts/play.py --mode hybrid            # Mode hybride

Entrée des coups en notation UCI (ex: e2e4, g1f3, e1g1 pour roque)
"""

import sys
import os
import argparse
import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ChessAI
from src.engine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Jouer contre Chess AI")
    parser.add_argument("--color", choices=["white", "black"], default="white",
                        help="Votre couleur (défaut: white)")
    parser.add_argument("--depth", type=int, default=4, help="Profondeur Minimax")
    parser.add_argument("--mode", choices=["minimax", "rl", "hybrid"], default="hybrid")
    parser.add_argument("--q-table", type=str, default="data/models/q_table.pkl")
    parser.add_argument("--time-limit", type=float, default=10.0)
    return parser.parse_args()


def print_board(board: chess.Board):
    """Affiche l'échiquier en ASCII coloré."""
    print()
    rows = str(board).split('\n')
    for i, row in enumerate(rows):
        rank = 8 - i
        print(f"  {rank} │ {row}")
    print("    └────────────────")
    print("      a b c d e f g h")
    print()


def get_player_move(board: chess.Board) -> chess.Move:
    """Demande un coup au joueur humain."""
    legal_ucis = {m.uci() for m in board.legal_moves}
    while True:
        raw = input("  Votre coup (UCI, ex: e2e4) : ").strip().lower()
        if raw in ("quit", "exit", "q"):
            print("\n👋 Partie abandonnée.")
            sys.exit(0)
        if raw == "moves":
            print(f"  Coups légaux : {', '.join(sorted(legal_ucis))}")
            continue
        if raw in legal_ucis:
            return chess.Move.from_uci(raw)
        print(f"  ❌ Coup illégal. Tapez 'moves' pour voir les coups disponibles.")


def main():
    args = parse_args()
    evaluator = Evaluator()

    human_color = chess.WHITE if args.color == "white" else chess.BLACK
    ai_color    = chess.BLACK if human_color == chess.WHITE else chess.WHITE

    ai = ChessAI(
        mode=args.mode,
        minimax_depth=args.depth,
        minimax_time_limit=args.time_limit,
        q_table_path=args.q_table if os.path.exists(args.q_table) else None,
        color=ai_color,
    )

    board = chess.Board()

    print("\n" + "=" * 50)
    print("  ♟  CHESS AI — Partie interactive")
    print("=" * 50)
    print(f"  Vous     : {'Blancs ♔' if human_color == chess.WHITE else 'Noirs ♚'}")
    print(f"  IA       : {'Blancs ♔' if ai_color == chess.WHITE else 'Noirs ♚'} (mode={args.mode}, d={args.depth})")
    print("  Commandes : 'moves' = voir les coups | 'quit' = quitter")
    print("=" * 50)

    while not board.is_game_over():
        print_board(board)
        score = evaluator.evaluate(board)
        turn_name = "Blancs" if board.turn == chess.WHITE else "Noirs"
        print(f"  → {turn_name} à jouer | Évaluation : {score:+.0f} cp")

        if board.is_check():
            print("  ⚠️  ÉCHEC !")

        if board.turn == human_color:
            move = get_player_move(board)
        else:
            print("  🤖 L'IA réfléchit...")
            move = ai.choose_move(board)
            if move is None:
                break
            print(f"  ♟ IA joue : {move.uci()}")

        board.push(move)
        print()

    print_board(board)
    result = board.result()
    if board.is_checkmate():
        winner = "Noirs" if board.turn == chess.WHITE else "Blancs"
        print(f"\n  🏆 ÉCHEC ET MAT ! {winner} gagnent ! ({result})")
    elif board.is_stalemate():
        print(f"\n  🤝 PAT — Nulle ! ({result})")
    else:
        print(f"\n  Résultat : {result}")


if __name__ == "__main__":
    main()
