import time
import matplotlib.pyplot as plt
from src.core.env import Connect4Env
from src.agents.minmax import MinimaxAgent


def compare_depths(d1, d2, num_games=50):
    env = Connect4Env()
    results = {1: 0, -1: 0, "draw": 0}

    print(f"[*] Duel de profondeurs : Depth {d1} (Rouge) vs Depth {d2} (Jaune)")

    agent_p1 = MinimaxAgent(depth=d1, player=1)
    agent_p2 = MinimaxAgent(depth=d2, player=-1)

    for i in range(num_games):
        env.reset()
        done = False
        while not done:
            agent = agent_p1 if env.current_player == 1 else agent_p2
            action = agent.act(env.board)
            _, _, done, info = env.step(action)

        winner = info.get("winner", 0)
        results[winner if winner != 0 else "draw"] += 1
        if (i + 1) % 5 == 0: print(f"Partie {i + 1}/{num_games} terminée")

    save_depth_chart(results, d1, d2, num_games)


def save_depth_chart(results, d1, d2, total):
    labels = [f'Depth {d1}', f'Depth {d2}', 'Nuls']
    values = [results[1], results[-1], results["draw"]]
    colors = ['#dc3c3c', '#f0dc50', '#888888']

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    plt.title(f"Impact de la profondeur de calcul\n(Minimax {d1} vs Minimax {d2})")
    plt.ylabel("Nombre de victoires")

    for i, v in enumerate(values):
        plt.text(i, v + 0.5, f"{v / total:.1%}", ha='center')

    plt.savefig("benchmark_depth.png")
    print("\n[+] Graphique 'benchmark_depth.png' généré !")


if __name__ == "__main__":
    compare_depths(1, 4, num_games=20)  # 20 parties suffisent car le calcul est plus long