from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
from typing import Dict, Optional, Tuple

import numpy as np

from src.core.board import Board, Player


@dataclass
class MinimaxAgent:
    depth: int = 4
    player: Player = -1
    center_weight: int = 3

    # Cache (transposition table) : évite de recalculer les mêmes états
    # key = (board_bytes, depth, maximizing, current_player)
    cache: Dict[tuple, Tuple[Optional[int], float]] = field(default_factory=dict)

    def act(self, board: Board) -> int:
        # reset cache à chaque coup (simple et efficace)
        self.cache.clear()

        best_col, _ = self._minimax(
            board=board,
            depth=self.depth,
            alpha=-inf,
            beta=inf,
            maximizing=True,
            current_player=self.player,
        )
        if best_col is None:
            actions = board.valid_actions()
            return actions[0] if actions else 0
        return best_col

    def _minimax(
        self,
        board: Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        current_player: Player,
    ) -> Tuple[Optional[int], float]:

        key = (board.grid.tobytes(), depth, maximizing, current_player)
        if key in self.cache:
            return self.cache[key]

        done, winner = board.terminal_status()
        if done:
            if winner == self.player:
                res = (None, 1e9)
            elif winner is None:
                res = (None, 0.0)
            else:
                res = (None, -1e9)
            self.cache[key] = res
            return res

        if depth == 0:
            res = (None, self._score_position_fast(board, self.player))
            self.cache[key] = res
            return res

        valid_cols = board.valid_actions()
        if not valid_cols:
            res = (None, 0.0)
            self.cache[key] = res
            return res

        # Explore centre d'abord (meilleur pruning)
        valid_cols.sort(key=lambda c: abs(3 - c))

        if maximizing:
            best_score = -inf
            best_col: Optional[int] = None
            for col in valid_cols:
                child = board.apply_action(col, current_player)
                _, score = self._minimax(child, depth - 1, alpha, beta, False, -current_player)
                if score > best_score:
                    best_score = score
                    best_col = col
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
            res = (best_col, best_score)
            self.cache[key] = res
            return res

        else:
            best_score = inf
            best_col = None
            for col in valid_cols:
                child = board.apply_action(col, current_player)
                _, score = self._minimax(child, depth - 1, alpha, beta, True, -current_player)
                if score < best_score:
                    best_score = score
                    best_col = col
                beta = min(beta, best_score)
                if alpha >= beta:
                    break
            res = (best_col, best_score)
            self.cache[key] = res
            return res

    # -------- Heuristique rapide (peu de numpy dans les loops) --------
    def _score_position_fast(self, board: Board, player: Player) -> float:
        g = board.grid  # numpy array (6,7)
        score = 0.0

        # bonus centre
        center_col = g[:, 3].tolist()
        score += center_col.count(player) * self.center_weight

        # fonctions helper rapides
        def eval_window(w: list[int]) -> float:
            opp = -player
            cp = w.count(player)
            co = w.count(opp)
            ce = w.count(0)

            s = 0.0
            if cp == 4:
                s += 1000
            elif cp == 3 and ce == 1:
                s += 50
            elif cp == 2 and ce == 2:
                s += 10

            if co == 3 and ce == 1:
                s -= 80
            return s

        # horizontales
        for r in range(board.ROWS):
            row = g[r, :].tolist()
            for c in range(board.COLS - 3):
                score += eval_window(row[c:c+4])

        # verticales
        for c in range(board.COLS):
            col = g[:, c].tolist()
            for r in range(board.ROWS - 3):
                score += eval_window(col[r:r+4])

        # diag bas-droite
        for r in range(board.ROWS - 3):
            for c in range(board.COLS - 3):
                w = [int(g[r+i, c+i]) for i in range(4)]
                score += eval_window(w)

        # diag bas-gauche
        for r in range(board.ROWS - 3):
            for c in range(3, board.COLS):
                w = [int(g[r+i, c-i]) for i in range(4)]
                score += eval_window(w)

        return score
