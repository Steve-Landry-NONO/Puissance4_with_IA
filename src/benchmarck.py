import time
import matplotlib.pyplot as plt
from src.core.board import Board
from src.core.env import Connect4Env
from src.agents.random_agent import RandomAgent
from src.agents.minmax import MinimaxAgent


def run_benchmark(p1, p2, num_games=100):
    env = Connect4Env()
    results = {1: 0, -1: 0, "draw": 0}

    print(f"[*] Début du benchmark : {num_games} parties...")
    start_time = time.time()

    for i in range(num_games):
        env.reset()
        done = False
        while not done:
            agent = p1 if env.current_player == 1 else p2
            action = agent.act(env.board)

            # Sécurité anti-coup invalide
            if not env.board.is_valid_action(action):
                action = env.board.valid_actions()[0]

            _, _, done, info = env.step(action)

        winner = info.get("winner", 0)
        if winner == 1:
            results[1] += 1
        elif winner == -1:
            results[-1] += 1
        else:
            results["draw"] += 1

    duration = time.time() - start_time
    save_chart(results, num_games, p1, p2)
    print(f"Benchmark terminé en {duration:.2f}s. Graphique sauvegardé !")


def save_chart(results, total, p1, p2):
    labels = ['P1 (Rouge)', 'P2 (Jaune)', 'Nuls']
    values = [results[1], results[-1], results["draw"]]
    colors = ['#dc3c3c', '#f0dc50', '#888888']  # Couleurs RED et YELLOW du projet

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, values, color=colors)
    plt.title(f"Performance : {p1.__class__.__name__} vs {p2.__class__.__name__}\n(Total: {total} parties)")
    plt.ylabel("Nombre de victoires")

    # Ajout du texte sur les barres
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval / total:.1%}", ha='center')

    plt.savefig("benchmark_results.png")
    plt.close()


if __name__ == "__main__":
    # Test Minimax vs Random
    run_benchmark(MinimaxAgent(depth=2, player=1), RandomAgent(), num_games=100)