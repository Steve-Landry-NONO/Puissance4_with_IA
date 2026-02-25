import numpy as np
from src.core.board import Board

def test_draw_game():
    grid = np.array([
        [ 1, -1, -1, -1,  1,  1, -1],
        [ 1,  1,  1, -1, -1,  1,  1],
        [-1,  1, -1,  1, -1, -1, -1],
        [ 1, -1, -1,  1, -1,  1,  1],
        [ 1, -1,  1,  1,  1, -1,  1],
        [-1,  1, -1, -1, -1,  1, -1],
    ], dtype=np.int8)

    b = Board(grid=grid)

    # sécurité : on vérifie qu'il n'y a pas de gagnant
    assert b.check_winner() is None

    # nul attendu : grille pleine + pas de gagnant
    assert b.is_draw() is True

    done, winner = b.terminal_status()
    assert done is True
    assert winner is None

def test_empty_board_has_no_winner():
    b = Board.empty()
    assert b.check_winner() is None
    assert not b.is_draw()

def test_drop_piece_falls_to_bottom():
    b = Board.empty()
    b.grid.setflags(write=True)
    row = b.drop_piece_inplace(3, 1)
    assert row == 5
    assert b.grid[5, 3] == 1

def test_horizontal_win():
    b = Board.empty()
    b.grid.setflags(write=True)
    for c in range(4):
        b.drop_piece_inplace(c, 1)
    assert b.check_winner() == 1

def test_vertical_win():
    b = Board.empty()
    b.grid.setflags(write=True)
    for _ in range(4):
        b.drop_piece_inplace(0, -1)
    assert b.check_winner() == -1

def test_diagonal_win_down_right():
    b = Board.empty()
    b.grid.setflags(write=True)
    b.drop_piece_inplace(0, 1)
    b.drop_piece_inplace(1, -1); b.drop_piece_inplace(1, 1)
    b.drop_piece_inplace(2, -1); b.drop_piece_inplace(2, -1); b.drop_piece_inplace(2, 1)
    b.drop_piece_inplace(3, -1); b.drop_piece_inplace(3, -1); b.drop_piece_inplace(3, -1); b.drop_piece_inplace(3, 1)
    assert b.check_winner() == 1

def test_action_mask_blocks_full_column():
    b = Board.empty()
    b.grid.setflags(write=True)
    for _ in range(6):
        b.drop_piece_inplace(0, 1)

    mask = b.action_mask()
    assert not mask[0]
    assert int(mask.sum()) == 6
