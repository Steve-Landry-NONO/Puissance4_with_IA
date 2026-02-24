from __future__ import annotations

import random

from src.core.board import Board


class RandomAgent:
    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, board: Board) -> int:
        actions = board.valid_actions()
        if not actions:
            # Pas d'action possible (grille pleine)
            return 0
        return self._rng.choice(actions)
# random agent

