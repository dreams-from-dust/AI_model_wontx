import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import os

BG_VOID = "#060907"
GLASS_BG = "#0d1410"
BORDER_GOLD = "#b4975a"
TEXT_BRIGHT = "#f1f6f3"
TEXT_MUTED = "#557262"
GLOW_GREEN = "#10b981"
GLOW_RED = "#ef4444"

def calculate_metrics(df):
    df['dt'] = df['timestamp'].diff().fillna(0.01)
    df['dist'] = np.sqrt(df['x'].diff().fillna(0)**2 + df['y'].diff().fillna(0)**2)
    df['velocity'] = (df['dist'] / df['dt']).clip(0, 5000)
    df['jitter'] = df['velocity'].diff().abs().fillna(0)
    return df

def analyze_owner_profile():
    db_path = 'wontxai.db'
    if not os.path.exists(db_path):
        print(f"❌ Error: {db_path} not found.")
        return

    try:
        conn = sqlite3.connect(db_path)
        df_mouse = pd.read_sql_query("SELECT timestamp, x, y FROM movement_logs WHERE label='owner'", conn)
        df_keys = pd.read_sql_query("SELECT dwell_time FROM keyboard_logs WHERE label='owner'", conn)
        conn.close()
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return

    if df_mouse.empty:
        print("❌ Error: No movement data found.")
        return

    df_m = calculate_metrics(df_mouse)
    v_mean = df_m['velocity'].mean()
    v_std = df_m['velocity'].std()

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Google Sans", "Product Sans", "Segoe UI", "Inter", "Century Gothic", "Arial"],
        "axes.edgecolor": BORDER_GOLD,
        "axes.labelcolor": TEXT_BRIGHT,
        "xtick.color": TEXT_MUTED,
        "ytick.color": TEXT_MUTED,
        "text.color": TEXT_BRIGHT,
        "axes.titleweight": "bold",
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })

    # Setup Figure with constrained layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18), constrained_layout=True)
    
    # Adding global padding to the container
    fig.set_constrained_layout_pads(w_pad=0.2, h_pad=0.2, hspace=0., wspace=0.)
    
    fig.patch.set_facecolor(BG_VOID)
    
    ax1.hist(df_m['velocity'], bins=50, color=GLOW_GREEN, alpha=0.6, edgecolor=BG_VOID, label='Owner Data')
    ax1.axvline(v_mean + (2 * v_std), color=GLOW_RED, linestyle=':', linewidth=2, label='Anomaly Threshold')
    ax1.set_title("Velocity Vector Distribution", pad=15)
    ax1.set_xlabel("Velocity (Pixels / Sec)")
    ax1.set_ylabel("Occurrences")
    ax1.legend(facecolor=GLASS_BG, edgecolor=BORDER_GOLD, loc='upper right')

    sample_size = min(len(df_m), 300)
    ax2.plot(df_m['jitter'].values[:sample_size], color=BORDER_GOLD, linewidth=1.2, label="Stability Index")
    ax2.set_title("Temporal Stability Trace", pad=15)
    ax2.set_xlabel("Telemetry Index")
    ax2.set_ylabel("Jitter Magnitude")
    ax2.legend(facecolor=GLASS_BG, edgecolor=BORDER_GOLD, loc='upper right')

    if not df_keys.empty:
        k_mean = df_keys['dwell_time'].mean()
        ax3.hist(df_keys['dwell_time'], bins=30, color=GLOW_GREEN, alpha=0.6, edgecolor=BG_VOID, label='Dwell Data')
        ax3.set_title(f"Keyboard Dwell Rhythm (Avg: {k_mean:.4f}s)", pad=15)
        ax3.set_xlabel("Dwell Time (Seconds)")
        ax3.set_ylabel("Occurrences")
        ax3.legend(facecolor=GLASS_BG, edgecolor=BORDER_GOLD, loc='upper right')
    else:
        ax3.text(0.5, 0.5, "No Keyboard Data Captured", ha='center', va='center', color=TEXT_MUTED)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor(GLASS_BG)
        ax.grid(color=BORDER_GOLD, linestyle=':', alpha=0.15)
        for spine in ax.spines.values():
            spine.set_edgecolor(BORDER_GOLD)
            spine.set_linewidth(0.5)

    # Professional Title using modern sans-serif layout guidelines
    fig.suptitle('Wontx AI • Behavioral Telemetry Analysis', color=BORDER_GOLD, 
                 fontsize=18, fontweight='bold')

    plt.show()

if __name__ == "__main__":
    analyze_owner_profile()