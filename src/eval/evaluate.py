# Winrate, benchmarks
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from src.core.env import Connect4Env
from src.agents.random_agent import RandomAgent
from src.agents.minmax import MinimaxAgent


@dataclass
class Stats:
    p1_wins: int = 0
    p2_wins: int = 0
    draws: int = 0
    aborted: int = 0  # si on dépasse max_moves (ne devrait pas arriver)


def play_match(
    env: Connect4Env,
    agent_p1,
    agent_p2,
    episodes: int = 200,
    alternate_start: bool = True,
    max_moves: int = 42,
) -> Stats:
    stats = Stats()

    for i in range(episodes):
        starting_player = 1 if (not alternate_start or i % 2 == 0) else -1
        env.reset(starting_player=starting_player)

        done = False
        info = {}
        moves = 0

        while not done and moves < max_moves:
            if env.current_player == 1:
                action = agent_p1.act(env.board)
            else:
                action = agent_p2.act(env.board)

            _, _, done, info = env.step(action)
            moves += 1

        if not done:
            stats.aborted += 1
            continue

        winner = info.get("winner")
        if winner == 1:
            stats.p1_wins += 1
        elif winner == -1:
            stats.p2_wins += 1
        else:
            stats.draws += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--alternate-start", action="store_true")
    parser.add_argument("--no-alternate-start", dest="alternate_start", action="store_false")
    parser.set_defaults(alternate_start=True)
    args = parser.parse_args()

    env = Connect4Env()

    p1 = RandomAgent(seed=123)
    p2 = MinimaxAgent(depth=args.depth, player=-1)

    t0 = time.perf_counter()
    stats = play_match(env, p1, p2, episodes=args.episodes, alternate_start=args.alternate_start)
    t1 = time.perf_counter()

    total = stats.p1_wins + stats.p2_wins + stats.draws + stats.aborted
    duration = t1 - t0
    gps = total / duration if duration > 0 else 0.0

    print(f"Depth: {args.depth}")
    print(f"Episodes: {total} | Time: {duration:.2f}s | Games/s: {gps:.1f}")
    if stats.aborted:
        print(f"Aborted (max_moves reached): {stats.aborted}")

    played = stats.p1_wins + stats.p2_wins + stats.draws
    if played > 0:
        print(f"P1 wins: {stats.p1_wins} ({stats.p1_wins/played:.2%})")
        print(f"P2 wins: {stats.p2_wins} ({stats.p2_wins/played:.2%})")
        print(f"Draws  : {stats.draws} ({stats.draws/played:.2%})")


if __name__ == "__main__":
    main()
