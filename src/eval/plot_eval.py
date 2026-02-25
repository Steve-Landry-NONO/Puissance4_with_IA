import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "results/eval.csv"
OUT_PATH = "results/eval_plot.png"

df = pd.read_csv(CSV_PATH)

# Winrate en %
df["winrate_pct"] = df["winrate"] * 100

# Timestamp -> tri chronologique si présent
if "timestamp" in df.columns:
    df = df.sort_values("timestamp")

# Colonne "scenario" lisible
def make_scenario(row):
    opp = row["opponent"]
    alt = row.get("alternate_start", False)
    depth = row.get("depth", "")
    if opp == "minimax":
        return f"minimax(d={depth}) | alt_start={alt}"
    return f"random | alt_start={alt}"

df["scenario"] = df.apply(make_scenario, axis=1)

# ---- Plot par scenario (courbes séparées) ----
plt.figure()

for scenario, g in df.groupby("scenario"):
    # x = index (ordre des runs) ou timestamp si tu préfères
    x = g["timestamp"] if "timestamp" in g.columns else range(len(g))
    plt.plot(x, g["winrate_pct"], marker="o", label=scenario)

plt.ylabel("Winrate (%)")
plt.xlabel("Timestamp" if "timestamp" in df.columns else "Run index")
plt.title("DQN evaluation winrate (grouped by scenario)")
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PATH, dpi=200)
plt.show()

print(f"Saved plot to: {OUT_PATH}")

# ---- Petit tableau résumé utile pour le rapport ----
summary = (
    df.groupby("scenario")[["winrate_pct"]]
    .agg(["count", "mean", "min", "max"])
    .round(2)
)
print(summary)
