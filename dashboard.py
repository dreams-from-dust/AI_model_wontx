import customtkinter as ctk
import sqlite3
import time
import os
import random
import threading 
import ctypes 
import subprocess
import sys
import numpy as np
import operator

# Core Matplotlib components for high fidelity rendering
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as gridspec

from agent import WontxAIAgent

# Design System Configuration
ctk.set_appearance_mode("dark")

BG_VOID = "#060907"          # Pure dark space void background
GLASS_BG = "#0d1410"         # Semi translucent card body
BORDER_DARK = "#15241b"      # Card frame micro borders
BORDER_GOLD = "#b4975a"      # Champagne gold accent
TEXT_BRIGHT = "#f1f6f3"      # Off white jade highlights
TEXT_MUTED = "#557262"       # Muted slate green secondary
GLOW_GREEN = "#10b981"       # Secure active emerald
GLOW_AMBER = "#f59e0b"       # Cognitive drift warning
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
    def __init__(self, master, icon_type="shield", size=24, color=BORDER_GOLD, **kwargs):
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

    def set_color(self, new_color):
        self.color = new_color
        self.draw_icon()

    def draw_icon(self):
        self.delete("all")
        s = self.size
        c = self.color
        
        if self.icon_type == "shield":
            pts = [
                (s * 0.5, s * 0.1),
                (s * 0.85, s * 0.22),
                (s * 0.85, s * 0.55),
                (s * 0.5, s * 0.9),
                (s * 0.15, s * 0.55),
                (s * 0.15, s * 0.22)
            ]
            self.create_polygon(pts, fill="", outline=c, width=2)
            self.create_line(s * 0.5, s * 0.22, s * 0.5, s * 0.78, fill=c, width=1.5)
            
        elif self.icon_type == "activity":
            pts = [
                (s * 0.1, s * 0.5),
                (s * 0.3, s * 0.5),
                (s * 0.4, s * 0.2),
                (s * 0.5, s * 0.8),
                (s * 0.6, s * 0.4),
                (s * 0.7, s * 0.6),
                (s * 0.9, s * 0.5)
            ]
            self.create_line(pts, fill=c, width=2, joinstyle="round")

        elif self.icon_type == "brain":
            self.create_oval(s * 0.25, s * 0.25, s * 0.5, s * 0.75, outline=c, width=1.5)
            self.create_oval(s * 0.5, s * 0.25, s * 0.75, s * 0.75, outline=c, width=1.5)
            self.create_oval(s * 0.35, s * 0.35, s * 0.42, s * 0.42, fill=c, outline="")
            self.create_oval(s * 0.58, s * 0.55, s * 0.65, s * 0.62, fill=c, outline="")
            self.create_line(s * 0.4, s * 0.4, s * 0.6, s * 0.6, fill=c, width=1)

        elif self.icon_type == "chart":
            self.create_line(s * 0.15, s * 0.85, s * 0.85, s * 0.85, fill=c, width=1.5)
            self.create_rectangle(s * 0.25, s * 0.5, s * 0.38, s * 0.85, outline=c, width=1.5)
            self.create_rectangle(s * 0.45, s * 0.3, s * 0.58, s * 0.85, outline=c, width=1.5)
            self.create_rectangle(s * 0.65, s * 0.4, s * 0.78, s * 0.85, outline=c, width=1.5)


class EmbeddedDashboardPlot(ctk.CTkFrame):
    """
    Hardware accelerated analytical plot displaying real time biometric metrics
    and deviation waveforms with smooth anti aliased fills.
    """
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        self.figure = Figure(figsize=(5, 2.5), dpi=100, facecolor=GLASS_BG)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor(BG_VOID)

        self.confidence_history = []
        self.smoothing_window = 10
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color(BORDER_DARK)
        self.ax.spines['bottom'].set_color(BORDER_DARK)
        
        self.ax.tick_params(colors=TEXT_MUTED, labelsize=10)
        self.ax.grid(True, color=BORDER_DARK, linestyle=':', linewidth=0.5)
        
        self.x_data = list(range(40))
        self.y_data = [100.0] * 40
        
        self.line, = self.ax.plot(self.x_data, self.y_data, color=GLOW_GREEN, linewidth=2, antialiased=True)
        self.fill = self.ax.fill_between(self.x_data, self.y_data, color=GLOW_GREEN, alpha=0.1)
        self.ax.set_ylim(0, 105)
        self.ax.set_xlim(0, 39)
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def process_agent_decision(self, confidence):
        self.confidence_history.append(confidence)
        if len(self.confidence_history) > self.smoothing_window:
          
           self.confidence_history.pop(0)
    
        # Only react if the AVERAGE confidence over the last 10 samples is low
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
    
        if avg_confidence < 30:
           self.trigger_lockdown()


    def update_metrics(self, current_trust_value):
        self.y_data.append(current_trust_value)
        if len(self.y_data) > 40:
            self.y_data.pop(0)
            
        self.line.set_ydata(self.y_data)
        self.fill.remove()
        
        if current_trust_value > 60:
            color = GLOW_GREEN
        elif current_trust_value > 30:
            color = GLOW_AMBER
        else:
            color = GLOW_RED
            
        self.line.set_color(color)
        self.fill = self.ax.fill_between(self.x_data, self.y_data, color=color, alpha=0.08)
        
        self.ax.relim()
        self.canvas.draw_idle()


class WontxAIDashboard(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Wontx AI • Risk Security Control Panel")
        self.geometry("1340x860")
        self.configure(fg_color=BG_VOID)

        self.is_monitoring = False
        self.current_trust = 100.0
        self.agent = WontxAIAgent()
        self.last_mouse_time = None
        self.last_mouse_pos = (0, 0)
        self.last_velocity = 0.0
        self.mouse_velocity_buffer = []
        self.key_press_times = {}
        self.last_key_release_time = time.time()
        
        # VERIFICATION SYSTEM: Only lock after confirming multiple anomalies
        self.decision_history = []  # Track last 5 decisions: [(action, conf), ...]
        self.anomaly_streak = 0  # Count consecutive high-confidence anomalies

        self.setup_ui()

        from pynput import keyboard
        self.key_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self.key_listener.start()

        self.bind('<Motion>', self.record_movement)
        self.after(500, self.roll_chart_timeline)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=340)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Command Sidebar Navigator
        self.sidebar = ctk.CTkFrame(self, fg_color=GLASS_BG, corner_radius=0, border_color=BORDER_DARK, border_width=1)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)

        # Bottom Diagnostics Tray Frame
        self.sidebar_bottom = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.sidebar_bottom.pack(side="bottom", fill="x", padx=25, pady=(15, 25))

        self.brain_btn = ctk.CTkButton(
            self.sidebar_bottom, 
            text="View Cognitive Model", 
            command=self.launch_brain_inspector,
            fg_color="transparent", 
            border_color=BORDER_GOLD, 
            border_width=2,
            text_color=BORDER_GOLD,    
            font=("Google Sans", 14, "bold"),
            hover_color="#111c16", 
            height=46, 
            corner_radius=12
        )
        self.brain_btn.pack(side="bottom", fill="x", pady=(10, 0))

        self.analytics_btn = ctk.CTkButton(
            self.sidebar_bottom, text="View Biometric Diagnostics", command=self.launch_analytics,
            fg_color="transparent", border_color=BORDER_GOLD, border_width=2,
            text_color=BORDER_GOLD, font=("Google Sans", 14, "bold"),
            hover_color="#111c16", height=46, corner_radius=12
        )
        self.analytics_btn.pack(side="bottom", fill="x")

        # Top Section layout elements (Packed from the top)
        self.brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.brand_frame.pack(side="top", pady=(35, 10), fill="x", padx=30)
        
        self.brand_title = ctk.CTkLabel(self.brand_frame, text="Wontx AI", font=("Google Sans Medium", 30), text_color=BORDER_GOLD)
        self.brand_title.pack(anchor="w")

        self.status_subline = ctk.CTkFrame(self.brand_frame, fg_color="transparent")
        self.status_subline.pack(anchor="w", pady=(5, 0))
        
        self.beacon_status = ctk.CTkLabel(self.status_subline, text="●", font=("Arial", 11), text_color=GLOW_GREEN)
        self.beacon_status.pack(side="left", padx=(0, 6))

        self.brand_desc = ctk.CTkLabel(self.status_subline, text="Edge Behavioral Firewall", font=("Google Sans", 11, "bold"), text_color=TEXT_MUTED)
        self.brand_desc.pack(side="left")

        self.sep_line = ctk.CTkFrame(self.sidebar, height=1, fg_color="#14261b")
        self.sep_line.pack(side="top", fill="x", padx=25, pady=15)

        self.actions_hdr = ctk.CTkLabel(self.sidebar, text="Behavioral Shield Operator", font=("Google Sans", 11, "bold"), text_color=TEXT_MUTED)
        self.actions_hdr.pack(side="top", anchor="w", padx=30, pady=(5, 5))

        # Primary Action Triggers
        self.monitor_btn = ctk.CTkButton(
            self.sidebar, text="Engage Behavioral Shield", command=self.toggle_monitoring,
            fg_color=BORDER_GOLD, text_color=BG_VOID, font=("Google Sans", 14, "bold"),
            height=52, hover_color="#cfae65", corner_radius=12,
            border_width=2, border_color=BORDER_GOLD
        )
        self.monitor_btn.pack(side="top", fill="x", padx=25, pady=6)

        self.enroll_btn = ctk.CTkButton(
            self.sidebar, text="Launch Profile Lab", command=self.launch_collector,
            fg_color="#0e1713", border_color=BORDER_DARK, border_width=1.5,
            text_color=TEXT_BRIGHT, font=("Google Sans", 14, "bold"),
            hover_color="#182b20", height=46, corner_radius=12
        )
        self.enroll_btn.pack(side="top", fill="x", padx=25, pady=5)

        self.train_btn = ctk.CTkButton(
            self.sidebar, text="Reinforcement Optimizer", command=self.launch_trainer,
            fg_color="#0e1713", border_color=BORDER_DARK, border_width=1.5,
            text_color=TEXT_BRIGHT, font=("Google Sans", 14, "bold"),
            hover_color="#182b20", height=46, corner_radius=12
        )
        self.train_btn.pack(side="top", fill="x", padx=25, pady=5)

        # Metrology Configuration Grid Box
        self.metrology_box = ctk.CTkFrame(self.sidebar, fg_color="#060b08", border_color=BORDER_DARK, border_width=1, corner_radius=10)
        self.metrology_box.pack(side="top", fill="x", padx=25, pady=(15, 10))

        metrology_lbl = ctk.CTkLabel(self.metrology_box, text="Compiler Parameters", font=("Google Sans", 12, "bold"), text_color=BORDER_GOLD)
        metrology_lbl.pack(anchor="w", padx=18, pady=(12, 4))
        
        self.meta_states = ctk.CTkLabel(self.metrology_box, text="• Learned States: 11", font=("Google Sans", 13), text_color=TEXT_BRIGHT)
        self.meta_states.pack(anchor="w", padx=18, pady=2)
        
        self.meta_actions = ctk.CTkLabel(self.metrology_box, text="• Solver Output: Binary Alert", font=("Google Sans", 13), text_color=TEXT_BRIGHT)
        self.meta_actions.pack(anchor="w", padx=18, pady=(2, 12))

        # Push flexible space spacer
        self.sidebar_push = ctk.CTkLabel(self.sidebar, text="")
        self.sidebar_push.pack(side="top", expand=True, fill="both")

        # Main Cognitive Workspace
        self.workspace = ctk.CTkFrame(self, fg_color="transparent")
        self.workspace.grid(row=0, column=1, sticky="nsew", padx=35, pady=35)

        # Top Bar status pill info
        self.status_strip = ctk.CTkFrame(self.workspace, fg_color="transparent")
        self.status_strip.pack(side="top", fill="x", pady=(0, 20))

        self.panel_title = ctk.CTkLabel(self.status_strip, text="Behavioral Telemetry Matrix", font=("Google Sans Light", 26), text_color=TEXT_BRIGHT)
        self.panel_title.pack(side="left")

        self.status_pill = ctk.CTkFrame(self.status_strip, fg_color="#081811", corner_radius=12, height=32)
        self.status_pill.pack(side="right", padx=(10, 0))
        self.status_pill.pack_propagate(False)

        self.status_indicator = ctk.CTkLabel(self.status_pill, text="Shield Inert", font=("Google Sans", 11, "bold"), text_color=TEXT_MUTED)
        self.status_indicator.pack(padx=18, expand=True)

        # Event stream log dock (Bottom of workspace)
        self.log_dock = ctk.CTkFrame(self.workspace, fg_color="transparent")
        self.log_dock.pack(side="bottom", fill="x", pady=(20, 0))

        self.ld_header = ctk.CTkFrame(self.log_dock, fg_color="transparent")
        self.ld_header.pack(fill="x", pady=(0, 8))

        self.ld_title = ctk.CTkLabel(self.ld_header, text="System Event Stream", font=("Google Sans", 12, "bold"), text_color=BORDER_GOLD)
        self.ld_title.pack(side="left")

        self.ld_mode = ctk.CTkLabel(self.ld_header, text="Standby", font=("Google Sans", 11, "bold"), text_color=TEXT_MUTED)
        self.ld_mode.pack(side="right")

        self.log_box = ctk.CTkTextbox(
            self.log_dock, fg_color=GLASS_BG, font=("Google Sans", 13), text_color=TEXT_BRIGHT,
            border_color=BORDER_DARK, border_width=1, height=130, corner_radius=8
        )
        self.log_box.pack(fill="x")
        self.log_box.configure(state="disabled")

        # Telemetry plot card fills the exact middle area elegantly
        self.telemetry_card = ctk.CTkFrame(self.workspace, fg_color=GLASS_BG, border_color=BORDER_DARK, border_width=1, corner_radius=12)
        self.telemetry_card.pack(side="top", fill="both", expand=True)

        self.card_header = ctk.CTkFrame(self.telemetry_card, fg_color="transparent")
        self.card_header.pack(fill="x", padx=25, pady=(20, 10))

        self.card_lbl = ctk.CTkLabel(self.card_header, text="Temporal Profile Coupling Status", font=("Google Sans", 13, "bold"), text_color=TEXT_MUTED)
        self.card_lbl.pack(side="left")

        self.live_readout = ctk.CTkLabel(self.card_header, text="100.0%", font=("Google Sans", 16, "bold"), text_color=GLOW_GREEN)
        self.live_readout.pack(side="right")

        self.tc_footer = ctk.CTkFrame(self.telemetry_card, fg_color="#060b08", height=54, corner_radius=8, border_color=BORDER_DARK, border_width=1)
        self.tc_footer.pack(side="bottom", fill="x", padx=25, pady=(0, 25))
        self.tc_footer.pack_propagate(False)

        self.metric_desc_lbl = ctk.CTkLabel(
            self.tc_footer, text="System in standby. Engage shield to activate biometric threat verification.",
            font=("Google Sans", 13), text_color=TEXT_MUTED
        )
        self.metric_desc_lbl.pack(fill="both", expand=True, padx=15)

        self.plot_panel = EmbeddedDashboardPlot(self.telemetry_card)
        self.plot_panel.pack(side="top", fill="both", expand=True, padx=25, pady=(0, 15))


    def launch_collector(self):
        subprocess.Popen([sys.executable, "collector.py"])

    def launch_trainer(self):
        subprocess.Popen([sys.executable, "train.py"])

    def launch_analytics(self):
        subprocess.Popen([sys.executable, "analyzer.py"])

    def launch_brain_inspector(self):
        """Launches the cognitive model diagnostic tool."""
        subprocess.Popen([sys.executable, "check_brain.py"])


    def process_telemetry_queues(self):
        pass

    def roll_chart_timeline(self):
        """Continuously feeds a rolling live baseline trend to the plot to keep the dashboard responsive and active."""
        if self.is_monitoring:
            self.plot_panel.update_metrics(self.current_trust)
            self.live_readout.configure(text=f"{int(self.current_trust)}.0%")
        self.after(500, self.roll_chart_timeline)


    def log_event(self, msg, color=TEXT_BRIGHT):
        self.log_box.configure(state="normal")
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def toggle_monitoring(self):
        self.is_monitoring = not self.is_monitoring
        if self.is_monitoring:
            self.current_trust = 100.0
            self.last_mouse_time = None
            self.last_velocity = 0.0
            self.key_press_times.clear()
            self.decision_history.clear()  # Reset verification history
            self.anomaly_streak = 0
            self.monitor_btn.configure(
                text="Disengage Behavioral Shield", 
                fg_color=GLOW_RED, 
                border_color=GLOW_RED,
                hover_color="#dc2626", 
                text_color=TEXT_BRIGHT
            )
            self.status_pill.configure(fg_color="#24140a")
            self.status_indicator.configure(text="Active Shield Monitoring", text_color=GLOW_AMBER)
            self.ld_mode.configure(text="Monitoring Online", text_color=GLOW_GREEN)
            self.log_event("Behavioral shield initiated. Ingesting metric signals.")
        else:
            self._reset_ui_to_idle()

    def _reset_ui_to_idle(self):
        self.is_monitoring = False
        self.current_trust = 100.0
        self.last_mouse_time = None
        self.last_velocity = 0.0
        self.key_press_times.clear()
        self.decision_history.clear()  # Reset verification history
        self.anomaly_streak = 0
        self.monitor_btn.configure(
            text="Engage Behavioral Shield", 
            fg_color=BORDER_GOLD, 
            border_color=BORDER_GOLD,
            hover_color="#cfae65", 
            text_color=BG_VOID
        )
        self.status_pill.configure(fg_color="#081811")
        self.status_indicator.configure(text="Shield Inert", text_color=TEXT_MUTED)
        self.ld_mode.configure(text="Standby", text_color=TEXT_MUTED)
        self.update_trust_ui()

    def record_movement(self, event):
        now = time.time()
        if self.is_monitoring and self.last_mouse_time is not None:
            dt = now - self.last_mouse_time
            if dt > 0.01:
                dist = np.sqrt((event.x - self.last_mouse_pos[0])**2 + (event.y - self.last_mouse_pos[1])**2)
                vel = dist / dt
                self.mouse_velocity_buffer.append(vel)
                if len(self.mouse_velocity_buffer) > 6:
                    self.mouse_velocity_buffer.pop(0)

                if len(self.mouse_velocity_buffer) >= 3:
                    avg_vel = np.mean(self.mouse_velocity_buffer)
                    deviation = abs(vel - avg_vel)

                    # Treat small variations around the local velocity baseline as normal.
                    if vel < self.agent.mouse_threshold * 0.8 and deviation < self.agent.mouse_threshold * 0.25:
                        action, conf = 0, 20.0
                    else:
                        action, conf = self.agent.get_decision(vel, input_type="mouse")

                    self._process_ai_result(action, conf)
                self.last_velocity = vel
        self.last_mouse_time = now
        self.last_mouse_pos = (event.x, event.y)

    def _on_key_press(self, key):
        if not self.is_monitoring: return
        if key not in self.key_press_times:
            self.key_press_times[key] = time.time()

    def _on_key_release(self, key):
        if not self.is_monitoring: return
        
        if key in self.key_press_times:
            now = time.time()
            dwell = now - self.key_press_times[key]
            
            # Flight time is the interval between releases
            flight = 0.0
            if self.last_key_release_time is not None:
                flight = now - self.last_key_release_time
            
            # This now matches the expected input in agent.py
            action, conf = self.agent.get_decision((dwell, flight), input_type="keyboard")
            
            self._process_ai_result(action, conf)
            
            self.last_key_release_time = now
            self.key_press_times.pop(key, None)

        if not self.is_monitoring: return
        if key in self.key_press_times:
            now = time.time()
            dwell = now - self.key_press_times[key]
            flight = 0.0
            if self.last_key_release_time is not None:
                flight = now - self.last_key_release_time
            action, conf = self.agent.get_decision((dwell, flight), input_type="keyboard")
            self._process_ai_result(action, conf)
            self.last_key_release_time = now
            self.key_press_times.pop(key, None)

    def _process_ai_result(self, action, confidence, state=None):
        """
        Verification System: Only penalize after confirming 4+ anomalies in last 5 events.
        This prevents false positives from locking the device.
        """
        # Track this decision in history (keep last 5)
        self.decision_history.append((action, confidence))
        if len(self.decision_history) > 5:
            self.decision_history.pop(0)
        
        # Count high-confidence anomalies in recent history
        recent_anomalies = sum(1 for a, c in self.decision_history if a == 1 and c > 80.0)
        
        # DEBUG: Show decisions being made
        if action == 1 or confidence > 50:
            self.log_event(f"[DEBUG] Action={action}, Conf={confidence:.0f}%, Anomalies={recent_anomalies}/5", TEXT_MUTED)
        
        # APPLY PENALTY ONLY IF 4+ ANOMALIES DETECTED IN LAST 5 EVENTS
        if recent_anomalies >= 4:
            # This is a confirmed pattern, not a single false positive
            penalty = 1.5  # Reduced penalty
            self.current_trust -= penalty
            self.anomaly_streak += 1
            self.log_event(f"CRITICAL: {recent_anomalies} anomalies confirmed. Trust: {self.current_trust:.1f}", GLOW_RED)
        else:
            # Either no anomalies or too few to confirm - FAST trust recovery
            recovery = 1.5  # Much faster recovery
            self.current_trust += recovery
            if action == 0:
                self.anomaly_streak = 0  # Reset streak on normal behavior

        self.current_trust = np.clip(self.current_trust, 0.0, 100.0)
        self.update_trust_ui()

        if self.current_trust <= 30:
            self.trigger_lockdown()
        elif self.current_trust < 60:
            if random.random() < 0.02:  # Reduced spam
                self.log_event("Warning: Behavioral drift threshold approaching.", GLOW_AMBER)

    def update_trust_ui(self):
        val = max(0, int(self.current_trust))
        self.plot_panel.update_metrics(val)
        self.live_readout.configure(text=f"{val}.0%")
        
        # --- Update this logic block to your preferred thresholds ---
        if val >= 75:  # Green/Secure Zone
            self.metric_desc_lbl.configure(text="Verification Matching: Biometrics Correlate With Owner Profile Signature", text_color=GLOW_GREEN)
            self.live_readout.configure(text_color=GLOW_GREEN)
            self.status_pill.configure(fg_color="#081811")
            self.status_indicator.configure(text="System Secure", text_color=GLOW_GREEN)
            
        elif 30 <= val < 75:  # Amber/Warning Zone
            self.metric_desc_lbl.configure(text="Warning: Behavioral Drift Threshold Breached", text_color=GLOW_AMBER)
            self.live_readout.configure(text_color=GLOW_AMBER)
            self.status_pill.configure(fg_color="#24140a")
            self.status_indicator.configure(text="Behavioral Divergence Warning", text_color=GLOW_AMBER)
            
        else:  # Red/Lockout Zone (val < 30)
            self.metric_desc_lbl.configure(text="Lockout Tripped: Intruder Profile Threshold Reached", text_color=GLOW_RED)
            self.live_readout.configure(text_color=GLOW_RED)
            self.status_pill.configure(fg_color="#2a080c")
            self.status_indicator.configure(text="Critical Anomaly Detected", text_color=GLOW_RED)

    def trigger_lockdown(self):
        # Prevent re-triggering if already locked
        if not self.is_monitoring:
            return
            
        self.is_monitoring = False
        self.log_event("CRITICAL: LOCKOUT ENGAGED", GLOW_RED)
        try:
            ctypes.windll.user32.LockWorkStation()
        except Exception as e:
            logging.error(f"Failed to lock: {e}")
        # Only run resume if specifically testing/demonstrating
        threading.Thread(target=self._demo_resume, daemon=True).start()

    def _demo_resume(self):
        time.sleep(5)
        self.after(0, self._reset_ui_to_idle)

    def on_closing(self):
        self.key_listener.stop()
        self.destroy()


if __name__ == "__main__":
    app = WontxAIDashboard()
    app.mainloop()
