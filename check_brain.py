import numpy as np
import matplotlib.pyplot as plt
import os
from agent import WontxAIAgent 

# Design System Config
BG_VOID = "#060907"
GLASS_BG = "#0d1410"
BORDER_GOLD = "#b4975a"
TEXT_BRIGHT = "#f1f6f3"
TEXT_MUTED = "#557262"
GLOW_GREEN = "#10b981"
GLOW_RED = "#ef4444"

def inspect_brain():
    model_path = "q_table.npy"
    if not os.path.exists(model_path):
        print(f"❌ Error: Q Table Model File '{model_path}' Not Found.")
        return

    agent = WontxAIAgent()
    q_table = agent.q_table 

    # Advanced UI Settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Google Sans", "Product Sans", "Segoe UI", "Inter", "Arial"],
        "axes.edgecolor": BORDER_GOLD,
        "axes.labelcolor": TEXT_BRIGHT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "text.color": TEXT_BRIGHT,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9
    })

    fig = plt.figure(figsize=(12, 10), facecolor=BG_VOID)
    
    # 1. Heatmap: Q Table Knowledge
    ax1 = fig.add_subplot(211, facecolor=GLASS_BG)
    im = ax1.imshow(q_table.T, cmap="RdYlGn", aspect="auto", interpolation="nearest")
    ax1.set_title("Behavioral Knowledge Map: Q-Table Weights", color=GLOW_GREEN, pad=20, fontsize=14, weight="bold")
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["Trust", "Alert"], weight="bold")
    ax1.set_xlabel("Behavioral State Severity (0 to 11)", fontsize=10)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.set_label("Reward Weight Magnitude", color=TEXT_BRIGHT)

    # 2. Confidence Distribution Density
    ax2 = fig.add_subplot(212, facecolor=GLASS_BG)
    samples = 1000
    test_values = np.random.uniform(0, 2500, samples)
    results = [agent.get_decision(val, input_type="mouse") for val in test_values]
    actions = np.array([r[0] for r in results])
    confidences = np.array([r[1] for r in results])

    ax2.hist(confidences[actions == 0], bins=30, color=GLOW_GREEN, alpha=0.3, label="Trust (Validated)", edgecolor=GLOW_GREEN)
    ax2.hist(confidences[actions == 1], bins=30, color=GLOW_RED, alpha=0.3, label="Alert (Anomaly)", edgecolor=GLOW_RED)
    
    ax2.set_title("Decision Confidence Density", color=BORDER_GOLD, pad=15, fontsize=12)
    ax2.set_xlabel("Confidence %", fontsize=10)
    ax2.set_ylabel("Occurrences", fontsize=10)
    ax2.legend(facecolor=GLASS_BG, edgecolor=BORDER_GOLD)

    plt.tight_layout(pad=4.0)
    plt.show()

if __name__ == "__main__":
    inspect_brain()