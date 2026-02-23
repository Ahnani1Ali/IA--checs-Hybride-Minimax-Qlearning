#!/usr/bin/env python3
"""
scripts/benchmark.py — Tournoi automatique entre configurations d'agents.

Usage :
    python scripts/benchmark.py --games 20
    python scripts/benchmark.py --games 50 --depths 2 3 4
"""

import sys
import os
import argparse
import chess
import json
import time
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import ChessAI


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Chess AI")
    parser.add_argument("--games", type=int, default=20, help="Parties par paire")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument("--output", type=str, default="data/benchmark_results.json")
    return parser.parse_args()


def play_match(depth_white: int, depth_black: int,
               n_games: int, time_limit: float) -> dict:
    """Joue n_games parties et retourne les résultats."""
    wins = draws = losses = 0
    total_moves = 0

    for g in range(n_games):
        white = ChessAI(mode='minimax', minimax_depth=depth_white,
                        minimax_time_limit=time_limit, color=chess.WHITE)
        black = ChessAI(mode='minimax', minimax_depth=depth_black,
                        minimax_time_limit=time_limit, color=chess.BLACK)
        result = white.play_game(opponent=black, max_moves=150, verbose=False)
        total_moves += len(result['moves'])

        if result['result'] == '1-0':
            wins += 1
        elif result['result'] == '0-1':
            losses += 1
        else:
            draws += 1

    return {
        'wins': wins, 'draws': draws, 'losses': losses,
        'win_rate': wins / n_games,
        'avg_moves': total_moves / n_games,
    }


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("=" * 65)
    print("  ♟  CHESS AI — Benchmark Tournoi")
    print("=" * 65)
    print(f"  Parties/paire : {args.games} | Profondeurs : {args.depths}")
    print(f"  Limite temps  : {args.time_limit}s")
    print("=" * 65)
    print(f"  {'Blanc':>12} {'Noir':>12} │ {'V':>5} {'N':>5} {'D':>5} │ {'WR%':>6} │ {'Moy.':>6}")
    print("  " + "─" * 63)

    all_results = []
    pairs = list(combinations(args.depths, 2)) + [(d, d) for d in args.depths]

    for d_w, d_b in pairs:
        label_w = f"Minimax-d{d_w}"
        label_b = f"Minimax-d{d_b}"
        start = time.time()
        res = play_match(d_w, d_b, args.games, args.time_limit)
        elapsed = time.time() - start
        wr = res['win_rate'] * 100

        print(f"  {label_w:>12} {label_b:>12} │ "
              f"{res['wins']:>5} {res['draws']:>5} {res['losses']:>5} │ "
              f"{wr:>5.1f}% │ "
              f"{res['avg_moves']:>5.0f}m  ({elapsed:.0f}s)")

        all_results.append({
            'white': label_w, 'black': label_b,
            'depth_white': d_w, 'depth_black': d_b,
            **res
        })

    print("=" * 65)

    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Résultats sauvegardés → {args.output}")


if __name__ == "__main__":
    main()
