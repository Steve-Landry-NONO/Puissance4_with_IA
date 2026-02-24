from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple
import numpy as np

Player = int  # convention: 1 et -1, vide = 0


@dataclass(frozen=False)
class Board:
    # --- Définition des dimensions et constantes ---
    # Ces valeurs sont utilisées par l'UI et la logique de victoire
    ROWS: int = 6
    COLS: int = 7
    WIN_LEN: int = 4

    EMPTY: int = 0
    P1: int = 1
    P2: int = -1

    grid: np.ndarray = None

    def __init__(self, grid=None):
        # Initialisation de la grille
        if grid is None:
            self.grid = np.zeros((self.ROWS, self.COLS), dtype=int)
        else:
            self.grid = np.array(grid, dtype=int)

    @staticmethod
    def empty() -> "Board":
        """Crée une instance de plateau vide."""
        return Board()

    def copy(self) -> "Board":
        """Crée une copie profonde du plateau pour les simulations de l'IA."""
        return Board(grid=self.grid.copy())

    # ---------- Validité des actions ----------
    def is_valid_action(self, col: int) -> bool:
        """Vérifie si une colonne est jouable (dans les limites et non pleine)."""
        return 0 <= col < self.COLS and self.grid[0, col] == self.EMPTY

    def valid_actions(self) -> list[int]:
        """Retourne la liste des indices de colonnes jouables."""
        return [c for c in range(self.COLS) if self.is_valid_action(c)]

    def action_mask(self) -> np.ndarray:
        """Masque booléen pour le réseau de neurones (DQN)."""
        return np.array([self.is_valid_action(c) for c in range(self.COLS)], dtype=bool)

    def is_full(self) -> bool:
        """Vérifie si le plateau est totalement rempli."""
        return not self.action_mask().any()

    # ---------- Mécanique de chute ----------
    def next_open_row(self, col: int) -> int:
        """Trouve la première ligne vide en partant du bas pour une colonne donnée."""
        if not (0 <= col < self.COLS):
            raise ValueError(f"Colonne hors limites: {col}")

        for r in range(self.ROWS - 1, -1, -1):
            if self.grid[r, col] == self.EMPTY:
                return r

        raise ValueError(f"La colonne {col} est pleine.")

    def drop_piece_inplace(self, col: int, player: Player) -> int:
        """Place un pion directement sur la grille actuelle."""
        if player not in (self.P1, self.P2):
            raise ValueError("Le joueur doit être 1 ou -1.")
        row = self.next_open_row(col)
        self.grid[row, col] = np.int8(player)
        return row

    def apply_action(self, col: int, player: Player) -> "Board":
        """Version pour Minimax : renvoie un nouveau Board sans modifier l'actuel."""
        new_board = self.copy()
        new_board.drop_piece_inplace(col, player)
        return new_board

    # ---------- Conditions de victoire ----------
    def check_winner(self) -> Optional[Player]:
        """Vérifie si un joueur a aligné WIN_LEN pions."""
        g = self.grid
        # Directions: horizontale, verticale, diagonale montante, diagonale descendante
        directions: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

        for r in range(self.ROWS):
            for c in range(self.COLS):
                p = int(g[r, c])
                if p == self.EMPTY:
                    continue

                for dr, dc in directions:
                    if self._has_line_from(r, c, dr, dc, p):
                        return p
        return None

    def _has_line_from(self, r: int, c: int, dr: int, dc: int, player: Player) -> bool:
        """Algorithme de vérification d'alignement selon WIN_LEN."""
        # Calcul de la position du dernier pion potentiel
        end_r = r + (self.WIN_LEN - 1) * dr
        end_c = c + (self.WIN_LEN - 1) * dc

        # Vérification des limites du plateau
        if not (0 <= end_r < self.ROWS and 0 <= end_c < self.COLS):
            return False

        # Vérification de l'alignement
        for k in range(1, self.WIN_LEN):
            if int(self.grid[r + k * dr, c + k * dc]) != player:
                return False
        return True

    def is_draw(self) -> bool:
        """Vérifie s'il y a match nul."""
        return self.is_full() and self.check_winner() is None

    def terminal_status(self) -> Tuple[bool, Optional[Player]]:
        """Retourne (Est-ce fini ?, Gagnant)."""
        w = self.check_winner()
        if w is not None:
            return True, w
        if self.is_draw():
            return True, None
        return False, None

    # ---------- Utilitaires IA et Console ----------
    def to_channels(self, current_player: Player) -> np.ndarray:
        """Encode la grille pour le DQN (canaux joueur/adversaire)."""
        g = self.grid
        me = (g == current_player).astype(np.float32)
        opp = (g == -current_player).astype(np.float32)
        return np.stack([me, opp], axis=-1)

    def __str__(self) -> str:
        return str(self.grid)