import customtkinter as ctk
import sqlite3
import time
import os
import numpy as np
import operator

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

ctk.set_appearance_mode("dark")

BG_VOID = "#060907"          # Pure dark space void background
GLASS_BG = "#0d1410"         # Semi translucent card body
BORDER_DARK = "#15241b"      # Card frame micro borders
BORDER_GOLD = "#b4975a"      # Champagne gold accent
TEXT_BRIGHT = "#f1f6f3"      # Off white jade highlights
TEXT_MUTED = "#557262"       # Muted slate green secondary
GLOW_GREEN = "#10b981"       # Secure active emerald
GLOW_RED = "#ef4444"         # Anomaly breach alert


def resolve_bg_color(widget, fallback=GLASS_BG):
    """Recursively walks up parent hierarchy to resolve transparent values to hex colors."""
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
    """Programmatic, mathematically drawn vector icon engine. Zero external files or emojis."""
    def __init__(self, master, icon_type="cursor", size=24, color=BORDER_GOLD, **kwargs):
        super().__init__(
            master, 
            width=size, 
            height=size, 
            highlightthickness=0, 
            **kwargs
        )
        self.size = size
        self.icon_type = icon_type
        self.color = color
        
        self.configure(bg=resolve_bg_color(master))
        self.draw_icon()

    def draw_icon(self):
        self.delete("all")
        s = self.size
        c = self.color
        
        if self.icon_type == "cursor":
            pts = [
                (s * 0.25, s * 0.2),
                (s * 0.75, s * 0.45),
                (s * 0.5, s * 0.55),
                (s * 0.65, s * 0.85),
                (s * 0.55, s * 0.9),
                (s * 0.4, s * 0.6),
                (s * 0.25, s * 0.7)
            ]
            self.create_polygon(pts, fill="", outline=c, width=2)
            
        elif self.icon_type == "keyboard":
            self.create_rectangle(s * 0.15, s * 0.3, s * 0.85, s * 0.7, outline=c, width=2)
            for i in range(1, 5):
                x = s * 0.15 + (s * 0.14 * i)
                self.create_line(x, s * 0.4, x, s * 0.6, fill=c, width=1.5)
            self.create_line(s * 0.3, s * 0.55, s * 0.7, s * 0.55, fill=c, width=2)

        elif self.icon_type == "lab":
            self.create_oval(s * 0.35, s * 0.15, s * 0.65, s * 0.45, outline=c, width=2)
            self.create_line(s * 0.5, s * 0.45, s * 0.5, s * 0.85, fill=c, width=2)
            self.create_oval(s * 0.2, s * 0.55, s * 0.8, s * 0.85, outline=c, width=1.5)


class EmbeddedCalibrationPlot(ctk.CTkFrame):
    """
    Renders live biometric calibration signal patterns (mouse and key counts) 
    using high fidelity scatter curves. Optimized height ratios to prevent spacing cuts.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        self.figure = Figure(figsize=(5, 2.5), dpi=100, facecolor=GLASS_BG)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(BG_VOID)
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color(BORDER_DARK)
        self.ax.spines['bottom'].set_color(BORDER_DARK)
        
        self.ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        self.ax.grid(True, color=BORDER_DARK, linestyle=':', linewidth=0.5)
        
        self.mouse_timeline = []
        self.key_timeline = []
        
        self.m_line, = self.ax.plot([], [], color=BORDER_GOLD, label="Mouse Inputs", linewidth=2, marker="o", markersize=4)
        self.k_line, = self.ax.plot([], [], color=GLOW_GREEN, label="Keyboard Inputs", linewidth=2, marker="s", markersize=4)
        
        self.ax.legend(facecolor=GLASS_BG, edgecolor=BORDER_DARK, labelcolor=TEXT_BRIGHT, fontsize=10)
        self.ax.set_ylim(0, 50)
        self.ax.set_xlim(0, 30)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def append_data(self, mouse_count, key_count):
        self.mouse_timeline.append(mouse_count)
        self.key_timeline.append(key_count)
        
        if len(self.mouse_timeline) > 30:
            self.mouse_timeline.pop(0)
            self.key_timeline.pop(0)
            
        x_steps = list(range(len(self.mouse_timeline)))
        
        self.m_line.set_data(x_steps, self.mouse_timeline)
        self.k_line.set_data(x_steps, self.key_timeline)
        
        max_limit = max(max(self.mouse_timeline, default=1), max(self.key_timeline, default=1)) + 10
        self.ax.set_ylim(0, max_limit)
        self.ax.set_xlim(0, max(30, len(x_steps)))
        
        self.ax.relim()
        self.canvas.draw_idle()


class WontxAICollector(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Wontx AI • Behavioral Profile Acquisition Lab")
        self.geometry("1080x840")
        self.configure(fg_color=BG_VOID)

        self.mouse_data = []
        self.key_data = []
        self.is_recording = False
        self.sample_goal = 5000

        self.key_press_times = {}
        self.last_key_release_time = None

        self.setup_ui()

        self.bind("<Motion>", self.record_movement)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)


    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.grid(row=0, column=0, sticky="ew", padx=45, pady=(35, 10))

        self.header_left = ctk.CTkFrame(self.header, fg_color="transparent")
        self.header_left.pack(side="left")

        self.lab_icon = VectorIcon(self.header_left, icon_type="lab", size=28, color=BORDER_GOLD)
        self.lab_icon.pack(side="left", padx=(0, 10))

        self.title_lbl = ctk.CTkLabel(
            self.header_left, text="Behavioral Profile Lab", 
            font=("Google Sans Light", 26), text_color=BORDER_GOLD
        )
        self.title_lbl.pack(side="left")

        self.sub_title_lbl = ctk.CTkLabel(
            self.header, text="Metric Acquisition Matrix Version 1.0", 
            font=("Google Sans", 12, "bold"), text_color=TEXT_MUTED
        )
        self.sub_title_lbl.pack(side="right", pady=(14, 0))


        self.main_card = ctk.CTkFrame(
            self, fg_color=GLASS_BG, border_color=BORDER_DARK, border_width=1, corner_radius=12
        )
        self.main_card.grid(row=1, column=0, sticky="nsew", padx=45, pady=(15, 35))

        # We arrange elements inside main_card using top and bottom layout priority
        # This guarantees buttons and consoles packed at the bottom remain perfectly visible
        self.desc_lbl = ctk.CTkLabel(
            self.main_card,
            text="Natural workspace interactions seed the behavioral matrices. Perform standard typing and navigation workflows to calibrate profile parameters.",
            font=("Google Sans", 15), text_color=TEXT_MUTED, wraplength=800
        )
        self.desc_lbl.pack(side="top", pady=(25, 15), padx=40)

        # Bottom packed elements (allocated first, bottom up)
        self.btn_frame = ctk.CTkFrame(self.main_card, fg_color="transparent")
        self.btn_frame.pack(side="bottom", pady=(20, 25))

        self.start_btn = ctk.CTkButton(
            self.btn_frame, text="Begin Acquisition", command=self.toggle_recording,
            font=("Google Sans", 15, "bold"), fg_color=BORDER_GOLD, text_color=BG_VOID,
            hover_color="#cfae65", height=52, width=260, corner_radius=12,
            border_width=2, border_color=BORDER_GOLD
        )
        self.start_btn.pack(side="left", padx=12)

        self.save_btn = ctk.CTkButton(
            self.btn_frame, text="Commit Profile Baseline", command=self.save_to_sqlite,
            font=("Google Sans", 15, "bold"), fg_color="transparent", border_color=BORDER_GOLD,
            border_width=2, text_color=BORDER_GOLD, hover_color="#122218",
            height=52, width=260, corner_radius=12
        )
        self.save_btn.pack(side="left", padx=12)

        self.console_card = ctk.CTkFrame(self.main_card, fg_color="#060b08", height=54, corner_radius=8, border_color=BORDER_DARK, border_width=1)
        self.console_card.pack(side="bottom", fill="x", padx=40, pady=5)
        self.console_card.pack_propagate(False)

        self.stats_lbl = ctk.CTkLabel(
            self.console_card, text="Ingest Console Standby • Calibration Labs Dispatched", 
            font=("Google Sans", 13, "bold"), text_color=TEXT_BRIGHT
        )
        self.stats_lbl.pack(fill="both", expand=True)

        self.progress = ctk.CTkProgressBar(
            self.main_card, height=10, progress_color=BORDER_GOLD, fg_color="#060b08", corner_radius=4
        )
        self.progress.pack(side="bottom", pady=(0, 15), padx=40, fill="x")
        self.progress.set(0)

        self.prog_hdr = ctk.CTkFrame(self.main_card, fg_color="transparent")
        self.prog_hdr.pack(side="bottom", fill="x", padx=42, pady=(5, 4))

        self.prog_lbl = ctk.CTkLabel(self.prog_hdr, text="Metric Acquisition Compliance Status", font=("Google Sans", 12, "bold"), text_color=TEXT_MUTED)
        self.prog_lbl.pack(side="left")

        self.prog_val = ctk.CTkLabel(self.prog_hdr, text="0%", font=("Google Sans", 13, "bold"), text_color=BORDER_GOLD)
        self.prog_val.pack(side="right")

        # The plot panel takes up the exact remaining flexible center space
        self.plot_panel = EmbeddedCalibrationPlot(self.main_card)
        self.plot_panel.pack(side="top", fill="both", expand=True, padx=40, pady=(5, 15))

        from pynput import keyboard
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def _on_press(self, key):
        self.on_press(key)

    def _on_release(self, key):
        self.on_release(key)


    def toggle_recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.start_btn.configure(
                text="Pause Acquisition", 
                fg_color=GLOW_RED, 
                border_color=GLOW_RED,
                hover_color="#dc2626", 
                text_color=TEXT_BRIGHT
            )
            self.stats_lbl.configure(text="Acquisition Engine Online: Recording Biometric Data", text_color=BORDER_GOLD)
        else:
            self.start_btn.configure(
                text="Begin Acquisition", 
                fg_color=BORDER_GOLD, 
                border_color=BORDER_GOLD,
                hover_color="#cfae65", 
                text_color=BG_VOID
            )
            self.stats_lbl.configure(text="Acquisition Engine Paused", text_color=TEXT_MUTED)

    def record_movement(self, event):
        if self.is_recording:
            self.mouse_data.append((time.time(), event.x, event.y, 'owner'))
            self.update_ui_stats()

    def on_press(self, key):
        if self.is_recording:
            self.key_press_times[key] = time.time()


    def on_release(self, key):
        if self.is_recording and key in self.key_press_times:
            now = time.time()
            dwell = operator.sub(now, self.key_press_times[key])
            
            flight = 0
            if self.last_key_release_time:
                flight = operator.sub(now, self.last_key_release_time)
            
            self.key_data.append((now, dwell, flight, 'owner'))
            self.last_key_release_time = now
            del self.key_press_times[key]
            self.update_ui_stats()

    def update_ui_stats(self):
        m = len(self.mouse_data)
        k = len(self.key_data)
        progress_val = min(m / max(self.sample_goal, 1), k / max(self.sample_goal, 1))
        self.progress.set(progress_val)
        
        self.prog_val.configure(text=f"{int(progress_val * 100)}%")
        self.stats_lbl.configure(
            text=f"Metric Stream Enroute • Mouse Capture: [{m}/{self.sample_goal}] • Keyboard Capture: [{k}/{self.sample_goal}]",
            text_color=TEXT_BRIGHT
        )
        
        self.plot_panel.append_data(m, k)


    def save_to_sqlite(self):
        if len(self.mouse_data) < self.sample_goal or len(self.key_data) < self.sample_goal:
            self.stats_lbl.configure(
                text=f"Sync Error: Need at least {self.sample_goal} mouse and {self.sample_goal} keyboard samples to build owner model.",
                text_color=GLOW_RED
            )
            return

        conn = sqlite3.connect('wontxai.db')
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS movement_logs 
        (id INTEGER PRIMARY KEY, timestamp REAL, x INTEGER, y INTEGER, label TEXT)''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS keyboard_logs 
        (id INTEGER PRIMARY KEY, timestamp REAL, dwell_time REAL, flight_time REAL, label TEXT)''')

        cursor.executemany('INSERT INTO movement_logs (timestamp, x, y, label) VALUES (?, ?, ?, ?)', self.mouse_data)
        cursor.executemany('INSERT INTO keyboard_logs (timestamp, dwell_time, flight_time, label) VALUES (?, ?, ?, ?)', self.key_data)
        
        conn.commit()
        conn.close()
        self.stats_lbl.configure(text="Metric Cogent Buffer Synchronized • Workstation Baseline Established", text_color=GLOW_GREEN)

    def on_closing(self):
        self.listener.stop()
        self.destroy()


if __name__ == "__main__":
    app = WontxAICollector()
    app.mainloop()