# src/agents/dqn/trainer.py
from __future__ import annotations

import os
import time
import random
from dataclasses import dataclass

import numpy as np

from src.core.env import Connect4Env
from src.agents.dqn.agent import DQNAgent

# Imports robustes (selon comment tu as nommé tes fichiers)
try:
    from src.agents.random_agent import RandomAgent
except Exception:  # pragma: no cover
    from src.agents.random import RandomAgent  # si jamais

try:
    # souvent c'est src/agents/minmax.py (comme tu avais avant)
    from src.agents.minmax import MinimaxAgent
except Exception:  # pragma: no cover
    # parfois c'est minimax.py
    from src.agents.minimax import MinimaxAgent


@dataclass
class TrainConfig:
    episodes: int = 5000

    # DQN training mechanics
    warmup_steps: int = 2000         # nb de moves DQN avant d'apprendre
    train_every: int = 4             # train tous les N moves DQN
    batch_size: int = 128

    # Gameplay / robustness
    train_on_both_sides: bool = True

    # Opponents
    minimax_depth: int = 2
    minimax_prob_max: float = 0.6    # proba max d'avoir minimax comme adversaire
    opponent: str = "mixed"          # "random" | "minimax" | "mixed"

    # Evaluation during training
    eval_every: int = 200
    eval_episodes: int = 200

    # Saving
    save_path: str = "models/dqn_connect4.keras"


def _get_obs(board, perspective_player: int) -> np.ndarray:
    """
    Encode toujours le plateau du point de vue de 'perspective_player':
    canal 0 = mes pions, canal 1 = pions adverses
    """
    return board.to_channels(perspective_player).astype(np.float32)


def _masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    q = q_values.copy()
    q[~mask] = -1e9
    return int(np.argmax(q))


def eval_vs_random(dqn: DQNAgent, episodes: int = 200, alternate_start: bool = True, seed: int = 0) -> float:
    """
    Évaluation en greedy (pas d'epsilon), contre RandomAgent.
    Retourne winrate (wins / episodes) du DQN.
    """
    rng = random.Random(seed)
    env = Connect4Env()
    opp = RandomAgent(seed=seed)

    wins = 0
    for ep in range(episodes):
        # DQN est parfois 1 parfois -1 si alternate_start=True
        dqn_player = 1
        if alternate_start and (ep % 2 == 1):
            dqn_player = -1

        env.reset(starting_player=1)  # le jeu commence toujours par P1
        done = False

        # si DQN est joueur 2, l'adversaire joue d'abord
        if dqn_player == -1:
            a0 = opp.act(env.board)
            env.step(a0)

        while not done:
            # tour DQN
            obs = _get_obs(env.board, dqn_player)
            mask = env.board.action_mask()
            # greedy
            q = dqn.q(obs[None, ...], training=False).numpy()[0]
            a = _masked_argmax(q, mask)

            _, r1, done, _ = env.step(a)
            if done:
                if r1 > 0:
                    wins += 1
                break

            # tour adversaire
            ao = opp.act(env.board)
            _, r2, done, _ = env.step(ao)
            if done:
                # r2 est du point de vue de l'adversaire → du point de vue DQN c'est négatif
                if r2 < 0:
                    wins += 1  # si env renvoie déjà reward "global", ça reste cohérent
                # dans la majorité des env connect4, si l'adversaire gagne, DQN perd
                break

    return wins / episodes


def train_dqn(cfg: TrainConfig) -> None:
    os.makedirs(os.path.dirname(cfg.save_path) or ".", exist_ok=True)

    env = Connect4Env()
    dqn = DQNAgent()

    random_opp = RandomAgent(seed=0)
    minimax_opp = MinimaxAgent(depth=cfg.minimax_depth)

    t0 = time.time()

    def pick_opponent(ep: int):
        if cfg.opponent == "random":
            return random_opp
        if cfg.opponent == "minimax":
            return minimax_opp

        # mixed: curriculum progressif vers minimax
        progress = ep / max(1, cfg.episodes)
        p_minimax = min(cfg.minimax_prob_max, progress * cfg.minimax_prob_max)
        return minimax_opp if random.random() < p_minimax else random_opp

    for ep in range(1, cfg.episodes + 1):
        # DQN joue en 1 ou en -1
        dqn_player = random.choice([1, -1]) if cfg.train_on_both_sides else 1

        env.reset(starting_player=1)
        done = False
        opp = pick_opponent(ep)

        # si DQN est joueur 2 : l'opposant joue d'abord
        if dqn_player == -1:
            a0 = opp.act(env.board)
            env.step(a0)

        while not done:
            # --- état avant action DQN (toujours du point de vue DQN) ---
            s = _get_obs(env.board, dqn_player)
            mask = env.board.action_mask()

            # action epsilon-greedy via ton DQNAgent
            a = dqn.act(s, mask)

            # 1) DQN joue
            _, r1, done, _ = env.step(a)

            if done:
                # terminal après le move DQN
                s2 = _get_obs(env.board, dqn_player)
                mask2 = env.board.action_mask()
                dqn.buffer.add(s, a, float(r1), s2, True, mask2)
                break

            # 2) Opposant joue
            ao = opp.act(env.board)
            _, r2, done, _ = env.step(ao)

            # reward vu du point de vue DQN :
            # si env renvoie reward "pour le joueur qui joue", alors le reward de l'adversaire est à inverser
            r = float(r1 - r2)

            s2 = _get_obs(env.board, dqn_player)
            mask2 = env.board.action_mask()
            dqn.buffer.add(s, a, r, s2, bool(done), mask2)

            # entraînement (après warmup)
            if dqn.step_count >= cfg.warmup_steps and (dqn.step_count % cfg.train_every == 0):
                dqn.train_step(cfg.batch_size)

        # --- logs / eval ---
        if ep % cfg.eval_every == 0:
            wr = eval_vs_random(dqn, episodes=cfg.eval_episodes, alternate_start=True, seed=0) * 100.0
            elapsed = time.time() - t0
            print(
                f"Episode {ep}/{cfg.episodes} | steps={dqn.step_count} | eps={dqn.epsilon():.3f} "
                f"| eval_vs_random({cfg.eval_episodes})={wr:.1f}% | elapsed={elapsed:.1f}s",
                flush=True,
            )

    dqn.save(cfg.save_path)
    print(f"Saved model to: {cfg.save_path}", flush=True)
