"""
Gestion des entrées utilisateur (humain) dans l'UI Pygame.
"""
from __future__ import annotations

def col_from_mouse_x(mouse_x: int, square_size: int) -> int:
    """Convertit une coordonnée X (pixels) en index de colonne."""
    return int(mouse_x // square_size)