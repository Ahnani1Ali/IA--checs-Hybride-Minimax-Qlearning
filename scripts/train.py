#!/usr/bin/env python3
"""
scripts/train.py — Script d'entraînement du Q-Learning par self-play.

Usage :
    python scripts/train.py --episodes 2000 --save data/models/q_table.pkl
    python scripts/train.py --episodes 5000 --verbose-every 200 --resume data/models/q_table.pkl
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rl.q_learning import QLearningAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement Q-Learning Chess AI")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Nombre d'épisodes de self-play (défaut: 2000)")
    parser.add_argument("--save", type=str, default="data/models/q_table.pkl",
                        help="Chemin de sauvegarde de la Q-table")
    parser.add_argument("--resume", type=str, default=None,
                        help="Reprendre depuis une Q-table existante")
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--verbose-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("♟  CHESS AI — Entraînement Q-Learning")
    print("=" * 60)
    print(f"  Épisodes     : {args.episodes}")
    print(f"  α (lr)       : {args.alpha}")
    print(f"  γ (discount) : {args.gamma}")
    print(f"  ε initial    : {args.epsilon}")
    print(f"  Sauvegarde   : {args.save}")
    print("=" * 60)

    os.makedirs(os.path.dirname(args.save), exist_ok=True)

    agent = QLearningAgent(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        q_table_path=args.resume,
    )

    start_time = time.time()
    agent.train(n_episodes=args.episodes, verbose_every=args.verbose_every)
    elapsed = time.time() - start_time

    agent.save(args.save)

    stats = agent.export_stats()
    total = stats['wins'] + stats['draws'] + stats['losses']
    wr = stats['wins'] / total * 100 if total else 0

    print("\n" + "=" * 60)
    print("✅ Résumé de l'entraînement")
    print("=" * 60)
    print(f"  Durée totale    : {elapsed:.1f}s ({elapsed/args.episodes*1000:.1f}ms/ep)")
    print(f"  Victoires       : {stats['wins']} ({wr:.1f}%)")
    print(f"  Nulles          : {stats['draws']}")
    print(f"  Défaites        : {stats['losses']}")
    print(f"  États Q appris  : {stats['q_table_size']:,}")
    print(f"  ε final         : {stats['epsilon']:.4f}")
    print(f"  Sauvegardé →    : {args.save}")


if __name__ == "__main__":
    main()
