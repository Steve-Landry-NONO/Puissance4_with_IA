from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.core.env import Connect4Env
from src.agents.random_agent import RandomAgent
from src.agents.minmax import MinimaxAgent


@dataclass
class Stats:
    wins: int = 0
    losses: int = 0
    draws: int = 0


def dqn_act(model: tf.keras.Model, state: np.ndarray, mask: np.ndarray) -> int:
    """Action greedy (epsilon=0) avec masking des colonnes pleines."""
    q = model(np.expand_dims(state, axis=0), training=False).numpy()[0]
    q_masked = np.where(mask, q, -1e9)
    return int(np.argmax(q_masked))


def evaluate(
    model_path: str,
    episodes: int = 200,
    opponent: str = "random",
    depth: int = 2,
    alternate_start: bool = False,
    seed: int = 0,
) -> Stats:
    """
    Évalue un modèle DQN contre un adversaire.

    Par défaut (alternate_start=False), le DQN commence toujours (starting_player=1),
    ce qui est recommandé car ton encodage utilise to_channels(1).

    Pour RandomAgent, on fait varier le seed à chaque épisode (seed + i)
    pour éviter qu'un seul random déterministe biaise l'évaluation.
    """
    env = Connect4Env()
    model = tf.keras.models.load_model(model_path)

    minimax_opp = MinimaxAgent(depth=depth, player=-1) if opponent == "minimax" else None

    stats = Stats()

    for i in range(episodes):
        # Qui commence ?
        starting_player = (1 if (i % 2 == 0) else -1) if alternate_start else 1
        env.reset(starting_player=starting_player)

        # Opponent instance
        if opponent == "random":
            opp = RandomAgent(seed=seed + i)
        else:
            opp = minimax_opp

        done = False
        info = {}

        while not done:
            if env.current_player == 1:
                s = env.board.to_channels(1)
                mask = env.action_mask()
                a = dqn_act(model, s, mask)
                _, _, done, info = env.step(a)
            else:
                a_opp = opp.act(env.board)
                _, _, done, info = env.step(a_opp)

        w = info.get("winner")
        if w == 1:
            stats.wins += 1
        elif w == -1:
            stats.losses += 1
        else:
            stats.draws += 1

    return stats


def append_csv_row(csv_path: Path, row: dict) -> None:
    """Ajoute une ligne à un CSV (crée le fichier et le header si besoin)."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="models/dqn_connect4.keras")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--opponent", choices=["random", "minimax"], default="random")
    p.add_argument("--depth", type=int, default=2)

    # ✅ Option d'alternance start (par défaut OFF)
    p.add_argument("--alternate-start", action="store_true")

    # ✅ Seed de base (RandomAgent utilisera seed+i)
    p.add_argument("--seed", type=int, default=0)

    # ✅ CSV output
    p.add_argument("--out", type=str, default="results/eval.csv")

    args = p.parse_args()

    s = evaluate(
        model_path=args.model,
        episodes=args.episodes,
        opponent=args.opponent,
        depth=args.depth,
        alternate_start=args.alternate_start,
        seed=args.seed,
    )

    total = s.wins + s.losses + s.draws
    winrate = s.wins / total if total else 0.0
    lossrate = s.losses / total if total else 0.0
    drawrate = s.draws / total if total else 0.0

    print(f"Episodes: {total}")
    print(f"Opponent: {args.opponent}" + (f"(depth={args.depth})" if args.opponent == "minimax" else ""))
    print(f"Alternate start: {args.alternate_start}")
    print(f"Seed base: {args.seed}")
    print(f"Wins : {s.wins} ({winrate:.2%})")
    print(f"Loss : {s.losses} ({lossrate:.2%})")
    print(f"Draw : {s.draws} ({drawrate:.2%})")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "episodes": args.episodes,
        "opponent": args.opponent,
        "depth": args.depth if args.opponent == "minimax" else "",
        "alternate_start": args.alternate_start,
        "seed_base": args.seed,
        "wins": s.wins,
        "losses": s.losses,
        "draws": s.draws,
        "winrate": round(winrate, 6),
        "lossrate": round(lossrate, 6),
        "drawrate": round(drawrate, 6),
    }

    out_path = Path(args.out)
    append_csv_row(out_path, row)
    print(f"Saved results to: {out_path}")
