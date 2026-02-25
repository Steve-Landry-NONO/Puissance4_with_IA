from __future__ import annotations

from typing import Protocol

from src.core.board import Board


class Agent(Protocol):
    def act(self, board: Board) -> int:
        """Retourne une action (colonne) valide."""
        ...
#interface agent

