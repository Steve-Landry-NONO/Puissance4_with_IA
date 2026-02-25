import numpy as np
from src.core.env import Connect4Env

def test_reset_shape():
    env = Connect4Env()
    s = env.reset()
    assert s.shape == (6, 7, 2)

def test_invalid_action_ends_game():
    env = Connect4Env(invalid_action_penalty=-10.0)
    env.reset()
    # colonne invalide (hors limites)
    s2, r, done, info = env.step(99)
    assert done is True
    assert r == -10.0
    assert info["invalid_action"] is True

def test_action_mask_after_filling_column():
    env = Connect4Env()
    env.reset()
    # Remplir colonne 0
    for _ in range(6):
        # current_player alterne automatiquement après chaque coup valide
        env.step(0)
        if env.done:
            break
    mask = env.action_mask()
    assert mask[0] == False
