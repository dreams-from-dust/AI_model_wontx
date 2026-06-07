import customtkinter as ctk
import numpy as np
import threading
import time
import random

from environment import WontxAIEnv
from brain import WontxAIBrain

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ctk.set_appearance_mode("dark")

BG_VOID = "#060907"
GLASS_BG = "#0d1410"
BORDER_DARK = "#15241b"
BORDER_GOLD = "#b4975a"
TEXT_BRIGHT = "#f1f6f3"
TEXT_MUTED = "#557262"
GLOW_GREEN = "#10b981"
GLOW_RED = "#ef4444"

def resolve_bg_color(widget, fallback=GLASS_BG):
    curr = widget
    while curr:
        if hasattr(curr, "cget"):
            try:
                color = curr.cget("fg_color")
                if color and color != "transparent":
                    if isinstance(color, (list, tuple)):
                        return color[1]
                    return color
            except Exception:
                pass
        try:
            color = curr.cget("bg")
            if color and color != "transparent":
                return color
        except Exception:
            pass
        curr = curr.master
    return fallback

class VectorIcon(ctk.CTkCanvas):
    def __init__(self, master, icon_type="Brain", size=24, color=BORDER_GOLD, **kwargs):
        super().__init__(master, width=size, height=size, highlightthickness=0, **kwargs)
        self.size = size
        self.icon_type = icon_type
        self.color = color
        self.configure(bg=resolve_bg_color(master))
        self.draw_icon()

    def draw_icon(self):
        self.delete("all")
        s = self.size
        c = self.color
        if self.icon_type == "Brain":
            self.create_oval(s*0.25, s*0.25, s*0.5, s*0.75, outline=c, width=1.5)
            self.create_oval(s*0.5, s*0.25, s*0.75, s*0.75, outline=c, width=1.5)
        elif self.icon_type == "Network":
            self.create_oval(s*0.4, s*0.15, s*0.6, s*0.35, outline=c, width=1.5)
            self.create_oval(s*0.15, s*0.6, s*0.35, s*0.8, outline=c, width=1.5)
            self.create_oval(s*0.65, s*0.6, s*0.85, s*0.8, outline=c, width=1.5)

class LiveTrainingPlot(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.figure = Figure(figsize=(5, 2.5), dpi=100, facecolor=GLASS_BG)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(BG_VOID)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["left"].set_color(BORDER_DARK)
        self.ax.spines["bottom"].set_color(BORDER_DARK)
        self.ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        self.ax.grid(True, color=BORDER_DARK, linestyle=":", linewidth=0.5)

        self.episodes = []
        self.rewards = []
        self.line, = self.ax.plot([], [], color=BORDER_GOLD, linewidth=2, marker="o", markersize=4)
        
        # Ensure layout fits labels by applying tight_layout
        self.figure.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def push_update(self, ep, reward):
        self.episodes.append(ep)
        self.rewards.append(reward)
        if len(self.episodes) > 30:
            self.episodes.pop(0)
            self.rewards.pop(0)
        
        self.line.set_data(range(len(self.episodes)), self.rewards)
        self.ax.set_xlim(0, max(30, len(self.episodes)))
        self.ax.set_ylim(min(self.rewards, default=0) - 2, max(self.rewards, default=1) + 2)
        self.figure.tight_layout()
        self.canvas.draw_idle()

class WontxAITrainerUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.total_episodes = 5000
        self.is_training = False
        self.is_paused = False
        self.title("Wontx AI • Reinforcement Layer Laboratory")
        self.geometry("1080x840")
        self.configure(fg_color=BG_VOID)
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="ew", padx=45, pady=(35, 10))

        self.header_left = ctk.CTkFrame(self.header, fg_color="transparent")
        self.header_left.pack(side="left")

        VectorIcon(self.header_left, "Network", 28).pack(side="left", padx=(0, 10))
        ctk.CTkLabel(self.header_left, text="Reinforcement Layer Engine", font=("Google Sans Light", 26), text_color=BORDER_GOLD).pack(side="left")

        ctk.CTkLabel(self.header, text="Bellman Solver Model Core", font=("Google Sans", 12, "bold"), text_color=TEXT_MUTED).pack(side="right", pady=(14, 0))

        self.main_card = ctk.CTkFrame(self, fg_color=GLASS_BG, border_color=BORDER_DARK, border_width=1, corner_radius=12)
        self.main_card.grid(row=1, column=0, sticky="nsew", padx=45, pady=(15, 35))
        
        self.desc_lbl = ctk.CTkLabel(self.main_card, text="Advanced reinforcement engine processing deep state action matrices to optimize predictive accuracy for behavioral convergence.", font=("Google Sans", 15), text_color=TEXT_MUTED, wraplength=800)
        self.desc_lbl.pack(side="top", pady=(25, 15), padx=40)

        self.params_grid = ctk.CTkFrame(self.main_card, fg_color="transparent")
        self.params_grid.pack(side="top", padx=40, pady=10, fill="x")
        self.create_parameter_node(self.params_grid, 0, "Learning Rate Alpha", "0.20")
        self.create_parameter_node(self.params_grid, 1, "Discount Rate Gamma", "0.95")
        self.create_parameter_node(self.params_grid, 2, "Optimization Depth", "5,000")

        self.btn_frame = ctk.CTkFrame(self.main_card, fg_color="transparent")
        self.btn_frame.pack(side="bottom", pady=(20, 25))
        
        self.train_btn = ctk.CTkButton(self.btn_frame, text="Run Model Gradients", command=self.start_training_thread, font=("Google Sans", 15, "bold"), fg_color=BORDER_GOLD, text_color=BG_VOID, hover_color="#cfae65", height=52, width=260, corner_radius=12, border_width=2, border_color=BORDER_GOLD)
        self.train_btn.pack(side="left", padx=12)
        
        self.pause_btn = ctk.CTkButton(self.btn_frame, text="Pause Training", command=self.toggle_pause, font=("Google Sans", 15, "bold"), fg_color="transparent", border_color=BORDER_GOLD, border_width=2, text_color=BORDER_GOLD, hover_color="#122218", height=52, width=260, corner_radius=12)
        self.pause_btn.pack(side="left", padx=12)

        self.console_card = ctk.CTkFrame(self.main_card, fg_color="#060b08", height=54, corner_radius=8, border_color=BORDER_DARK, border_width=1)
        self.console_card.pack(side="bottom", fill="x", padx=40, pady=5)
        self.console_card.pack_propagate(False)
        self.stats_lbl = ctk.CTkLabel(self.console_card, text="Engine Standby • Optimization Labs Dispatched", font=("Google Sans", 13, "bold"), text_color=TEXT_BRIGHT)
        self.stats_lbl.pack(fill="both", expand=True)

        self.progress = ctk.CTkProgressBar(self.main_card, height=10, progress_color=BORDER_GOLD, fg_color="#060b08", corner_radius=4)
        self.progress.pack(side="bottom", pady=(0, 15), padx=40, fill="x")
        self.progress.set(0)

        self.plot_panel = LiveTrainingPlot(self.main_card)
        self.plot_panel.pack(side="top", fill="both", expand=True, padx=40, pady=(5, 15))

    def create_parameter_node(self, parent, col, title, value):
        parent.grid_columnconfigure(col, weight=1)
        card = ctk.CTkFrame(parent, fg_color="#060b08", border_color=BORDER_DARK, border_width=1, corner_radius=8)
        card.grid(row=0, column=col, padx=12, sticky="ew")
        ctk.CTkLabel(card, text=title, font=("Google Sans", 11, "bold"), text_color=TEXT_MUTED).pack(pady=(10, 2))
        ctk.CTkLabel(card, text=value, font=("Google Sans", 18, "bold"), text_color=TEXT_BRIGHT).pack(pady=(2, 10))

    def start_training_thread(self):
        if self.is_training: return
        self.is_training = True
        threading.Thread(target=self.run_training_logic, daemon=True).start()

    def toggle_pause(self):
        self.is_paused = not self.is_paused

    def run_training_logic(self):
        try:
            self.after(0, lambda: self.stats_lbl.configure(text="Initializing Training Environment...", text_color=TEXT_BRIGHT))
            env, brain = WontxAIEnv(), WontxAIBrain()
            self.after(0, lambda: self.stats_lbl.configure(text="Environment Ready • Beginning Training Loop", text_color=GLOW_GREEN))
            
            for e in range(self.total_episodes):
                while self.is_paused: time.sleep(0.1)
                
                # 1. Ensure state is an integer
                state = int(env.reset()) 
                done = False
                total_reward = 0
                
                while not done:
                    # 2. Brain expects an integer index
                    action = brain.get_action(state) 
    
                    # 3. Environment returns (next_state, reward, done)
                    next_state_raw, reward, done = env.step(state, action)
                    
                    # 4. Force next_state to integer
                    next_state = int(next_state_raw)
    
                    # 5. Update Brain
                    brain.update_q_table(state, action, reward, next_state)
                    
                    total_reward += reward
                    state = next_state
                
                ratio = (e + 1) / self.total_episodes
                self.after(0, lambda ep=e+1, r=total_reward, rt=ratio: self.update_ui(ep, r, rt))
            
            self.after(0, lambda: self.stats_lbl.configure(text="Training Complete • Q-Table Saved Successfully", text_color=GLOW_GREEN))
            self.is_training = False
            
        except Exception as e:
            # This will show you EXACTLY where the error is
            import traceback
            error_msg = traceback.format_exc()
            print(error_msg)
            self.after(0, lambda: self.stats_lbl.configure(text=f"Error: {str(e)}", text_color=GLOW_RED))
            self.is_training = False
            
    def update_ui(self, ep, r, ratio):
        self.progress.set(ratio)
        self.stats_lbl.configure(text=f"Optimization Cycle [{ep}] • Reward: {r:.2f} • Progress: {int(ratio*100)}%", text_color=TEXT_BRIGHT)
        self.plot_panel.push_update(ep, r)

if __name__ == "__main__":
    app = WontxAITrainerUI()
    app.mainloop()