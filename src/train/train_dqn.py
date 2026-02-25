# src/train/train_dqn.py
from src.agents.dqn.trainer import TrainConfig, train_dqn

if __name__ == "__main__":
    config = TrainConfig(
        episodes=5000,
        train_on_both_sides=True,
        opponent="mixed",
        minimax_depth=2,
        minimax_prob_max=0.6,
        eval_every=200,
        eval_episodes=200,
        save_path="models/dqn_connect4.keras",
    )
    train_dqn(config)
