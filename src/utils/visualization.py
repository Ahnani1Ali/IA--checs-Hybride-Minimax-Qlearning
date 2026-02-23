"""
visualization.py — Utilitaires d'affichage pour Jupyter Notebook.

Fonctions :
  - render_board()      : affiche l'échiquier en SVG dans Jupyter
  - plot_eval_curve()   : courbe d'évaluation au fil des coups
  - plot_training_rl()  : courbe d'entraînement RL (wins/losses)
  - print_opening_tree(): affiche l'arbre d'ouvertures
"""

import chess
import chess.svg
import io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from IPython.display import SVG, display, HTML
from typing import List, Optional, Dict


def render_board(board: chess.Board,
                 size: int = 400,
                 last_move: Optional[chess.Move] = None,
                 arrows: Optional[list] = None) -> SVG:
    """
    Affiche l'échiquier en SVG dans Jupyter.

    Parameters
    ----------
    board     : position actuelle
    size      : taille en pixels
    last_move : colorie le dernier coup joué
    arrows    : liste de (from_sq, to_sq, color) pour dessiner des flèches
    """
    svg_arrows = []
    if arrows:
        for arrow in arrows:
            if len(arrow) == 3:
                svg_arrows.append(chess.svg.Arrow(arrow[0], arrow[1], color=arrow[2]))
            elif len(arrow) == 2:
                svg_arrows.append(chess.svg.Arrow(arrow[0], arrow[1]))

    svg_str = chess.svg.board(
        board=board,
        size=size,
        lastmove=last_move,
        arrows=svg_arrows,
    )
    return SVG(svg_str)


def display_move_sequence(board_init: chess.Board, moves: List[str],
                          pause: float = 0) -> None:
    """Affiche la séquence de coups pas à pas."""
    board = board_init.copy()
    for i, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        print(f"Coup {i+1}: {uci}")
        board.push(move)
        display(render_board(board, last_move=move))


def plot_eval_curve(eval_scores: List[float],
                    moves: Optional[List[str]] = None,
                    title: str = "Courbe d'évaluation") -> plt.Figure:
    """
    Trace la courbe d'évaluation au fil de la partie.

    Parameters
    ----------
    eval_scores : liste de scores en centipawns
    moves       : liste des coups UCI correspondants
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    x = range(len(eval_scores))
    ax.plot(x, eval_scores, linewidth=2, color='#2196F3', label='Évaluation')
    ax.fill_between(x, eval_scores, 0,
                    where=[s >= 0 for s in eval_scores],
                    alpha=0.2, color='white', label='Avantage Blancs')
    ax.fill_between(x, eval_scores, 0,
                    where=[s < 0 for s in eval_scores],
                    alpha=0.3, color='#222222', label='Avantage Noirs')

    ax.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Numéro de demi-coup')
    ax.set_ylabel('Évaluation (centipawns)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if moves:
        ax.set_xticks(range(len(moves)))
        if len(moves) <= 30:
            ax.set_xticklabels(moves, rotation=45, fontsize=7)

    plt.tight_layout()
    return fig


def plot_training_progress(stats_history: List[Dict],
                           window: int = 50) -> plt.Figure:
    """
    Trace les métriques d'entraînement RL.

    Parameters
    ----------
    stats_history : liste de dicts {'wins', 'draws', 'losses'} par épisode
    window        : fenêtre de lissage (moving average)
    """
    wins   = [s.get('wins', 0) for s in stats_history]
    draws  = [s.get('draws', 0) for s in stats_history]
    losses = [s.get('losses', 0) for s in stats_history]
    epsilons = [s.get('epsilon', 0) for s in stats_history]

    def moving_avg(data, w):
        return np.convolve(data, np.ones(w)/w, mode='valid')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))

    episodes = range(len(stats_history))

    # Courbes Wins/Draws/Losses
    if len(wins) >= window:
        ax1.plot(range(window-1, len(wins)),
                 moving_avg(wins, window), color='#4CAF50', label='Victoires')
        ax1.plot(range(window-1, len(draws)),
                 moving_avg(draws, window), color='#FFC107', label='Nulles')
        ax1.plot(range(window-1, len(losses)),
                 moving_avg(losses, window), color='#F44336', label='Défaites')
    else:
        ax1.plot(episodes, wins, color='#4CAF50', label='Victoires')
        ax1.plot(episodes, draws, color='#FFC107', label='Nulles')
        ax1.plot(episodes, losses, color='#F44336', label='Défaites')

    ax1.set_title(f'Résultats par épisode (moyenne mobile {window})')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Nombre')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Epsilon decay
    ax2.plot(episodes, epsilons, color='#9C27B0', linewidth=2)
    ax2.set_title('Décroissance de ε (exploration)')
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('ε (epsilon)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_piece_heatmap(board: chess.Board, color: chess.Color = chess.WHITE) -> plt.Figure:
    """
    Carte de chaleur des pièces d'une couleur sur l'échiquier.
    """
    grid = np.zeros((8, 8))
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.color == color:
            row = chess.square_rank(square)
            col = chess.square_file(square)
            from chess_ai.src.engine.evaluator import PIECE_VALUES
            grid[row][col] = PIECE_VALUES.get(piece.piece_type, 0) / 100

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap='YlOrRd', origin='lower',
                   vmin=0, vmax=10, aspect='equal')

    ax.set_xticks(range(8))
    ax.set_xticklabels(['a','b','c','d','e','f','g','h'])
    ax.set_yticks(range(8))
    ax.set_yticklabels(['1','2','3','4','5','6','7','8'])

    color_str = 'Blancs' if color == chess.WHITE else 'Noirs'
    ax.set_title(f'Carte des pièces — {color_str}')
    plt.colorbar(im, ax=ax, label='Valeur (pions)')
    plt.tight_layout()
    return fig


def display_opening_name(board: chess.Board, opening_book) -> None:
    """Affiche le nom de l'ouverture actuelle."""
    name = opening_book.get_opening_name(board)
    html = f"""
    <div style="background:#1a1a2e; color:#e0e0e0; padding:10px;
                border-radius:8px; font-family:monospace; margin:5px 0">
        <b>♟ Ouverture :</b> {name} |
        <b>Demi-coups :</b> {board.fullmove_number}
    </div>
    """
    display(HTML(html))
