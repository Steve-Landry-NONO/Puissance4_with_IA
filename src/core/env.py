from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.core.board import Board, Player


@dataclass
class StepInfo:
    invalid_action: bool = False
    winner: Optional[Player] = None
    draw: bool = False


class Connect4Env:
    """
    Environnement Puissance 4 pour RL.
    - Action: colonne [0..6]
    - Observation: (6,7,2) canaux (joueur courant / adversaire)
    - Reward (du point de vue du joueur qui joue l'action):
        +1.0 victoire
        -1.0 défaite (si on laisse l'adversaire jouer ensuite et gagne)
        0.0 sinon
        (optionnel) -10.0 action invalide
    """

    def __init__(
        self,
        invalid_action_penalty: float = -10.0,
        win_reward: float = 1.0,
        lose_reward: float = -1.0,
        draw_reward: float = 0.0,
    ) -> None:
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.win_reward = float(win_reward)
        self.lose_reward = float(lose_reward)
        self.draw_reward = float(draw_reward)

        self.board: Board = Board.empty()
        self.current_player: Player = 1  # joueur qui doit jouer maintenant
        self.done: bool = False

    def reset(self, starting_player: Player = 1) -> np.ndarray:
        self.board = Board.empty()
        self.board.grid.setflags(write=True)  # autorise drop_piece_inplace
        self.current_player = starting_player
        self.done = False
        return self._obs()

    def action_mask(self) -> np.ndarray:
        return self.board.action_mask()

    def _obs(self) -> np.ndarray:
        return self.board.to_channels(self.current_player)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécute un coup pour current_player.

        Retour:
        - next_state (observation du prochain joueur)
        - reward (pour le joueur qui vient de jouer)
        - done
        - info dict (invalid_action, winner, draw)
        """
        if self.done:
            # Convention: si step() appelé après fin, on renvoie état inchangé
            info = StepInfo(invalid_action=False, winner=self.board.check_winner(), draw=self.board.is_draw())
            return self._obs(), 0.0, True, info.__dict__

        # Action invalide ?
        if not isinstance(action, (int, np.integer)) or not self.board.is_valid_action(int(action)):
            self.done = True  # on peut considérer fin immédiate en cas d'action invalide
            info = StepInfo(invalid_action=True, winner=-self.current_player, draw=False)
            # punition forte
            return self._obs(), self.invalid_action_penalty, True, info.__dict__

        action = int(action)
        self.board.drop_piece_inplace(action, self.current_player)

        # Check terminal après le coup
        winner = self.board.check_winner()
        if winner is not None:
            self.done = True
            info = StepInfo(invalid_action=False, winner=winner, draw=False)
            reward = self.win_reward if winner == self.current_player else self.lose_reward
            return self._next_turn_obs(), reward, True, info.__dict__

        if self.board.is_draw():
            self.done = True
            info = StepInfo(invalid_action=False, winner=None, draw=True)
            return self._next_turn_obs(), self.draw_reward, True, info.__dict__

        # Partie continue : on passe au joueur suivant
        self.current_player = -self.current_player
        return self._obs(), 0.0, False, StepInfo().__dict__

    def _next_turn_obs(self) -> np.ndarray:
        """
        Observation pour le joueur suivant (utile même si done=True, pour cohérence).
        """
        next_player = -self.current_player
        return self.board.to_channels(next_player)

    def observation(self) -> np.ndarray:
        return self._obs()
