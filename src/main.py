from __future__ import annotations
import argparse
import os
import sys
import numpy as np

from src.core.board import Board
from src.core.env import Connect4Env
from src.agents.humain import HumanAgent
from src.agents.random_agent import RandomAgent
from src.agents.minmax import MinimaxAgent
from src.ui.pygame_app import PygameApp

class KerasDQNPlayer:
    """Wrapper avec import local pour éviter les crashs de librairies graphiques."""
    def __init__(self, model_path: str):
        print("[*] Chargement de TensorFlow...")
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)

    def act(self, board: Board) -> int:
        import tensorflow as tf
        # Détection du joueur actuel pour l'encodage
        nb_pieces = np.count_nonzero(board.grid)
        current_player = 1 if nb_pieces % 2 == 0 else -1

        state = board.to_channels(current_player)
        mask = board.action_mask()
        q = self.model.predict(state[None, ...], verbose=0)[0]
        q = np.where(mask, q, -1e9)
        return int(np.argmax(q))

def build_agent(name: str, model_path: str, depth: int):
    name = name.lower()
    if name in ["human", "humain"]:
        return HumanAgent()
    if name == "random":
        return RandomAgent()
    if name == "minimax":
        return MinimaxAgent(depth=depth)
    if name == "dqn":
        return KerasDQNPlayer(model_path)
    raise ValueError(f"Agent inconnu: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p1", default="human", choices=["dqn", "human", "random", "minimax"])
    parser.add_argument("--p2", default="minimax", choices=["dqn", "human", "random", "minimax"])
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--model", type=str, default="models/dqn_connect4.keras")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--delay-ms", type=int, default=300)
    args = parser.parse_args()

    # Si on ne joue PAS avec un DQN, on ne charge jamais TensorFlow
    p1 = build_agent(args.p1, args.model, args.depth)
    p2 = build_agent(args.p2, args.model, args.depth)

    if args.headless or os.environ.get("SDL_VIDEODRIVER") == "dummy":
        env = Connect4Env()
        env.reset()
        print(f"[*] Match Headless: {args.p1} vs {args.p2}")
        # Logique simplifiée pour le test
        return

    # Lancement de l'UI
    print(f"[*] Démarrage UI: {args.p1} vs {args.p2}")
    board = Board(grid=np.zeros((6, 7), dtype=int))
    app = PygameApp(board, p1, p2, delay_ms=args.delay_ms)
    app.executer()
if __name__ == "__main__":
    main()