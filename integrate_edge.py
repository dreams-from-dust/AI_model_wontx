import pynput
from pynput import mouse, keyboard
import time
import numpy as np
from brain import WontxAIBrain   # FIXED: use ML model, not Q-table brain


class WontxAIGuard:
    """
    Edge Behavioral Security Agent
    Works as embedded authentication layer inside any app
    """

    def __init__(self, on_breach_callback=None):

        self.brain = WontxAIBrain()

        self.on_breach = on_breach_callback
        self.is_active = False

        # ===============================
        # TIME-WINDOW BASED MEMORY (KEY FIX)
        # ===============================
        self.mouse_buffer = []
        self.key_buffer = []

        self.window_size = 2.5  # seconds

        self.last_mouse_time = time.time()
        self.last_mouse_pos = (0, 0)

        self.key_press_times = {}

        # risk model (replaces trust_score)
        self.risk_score = 0.0

    # ===============================
    # START / STOP
    # ===============================
    def start_protection(self):
        self.is_active = True

        self.mouse_listener = mouse.Listener(on_move=self._process_mouse)
        self.key_listener = keyboard.Listener(
            on_press=self._process_keys,
            on_release=self._process_key_release
        )

        self.mouse_listener.start()
        self.key_listener.start()

        print("🛡️ Wontx AI Edge Guard ACTIVE")

    def stop_protection(self):
        self.is_active = False
        self.mouse_listener.stop()
        self.key_listener.stop()

    # ===============================
    # FEATURE EXTRACTION (IMPORTANT UPGRADE)
    # ===============================
    def _extract_features(self):
        """
        Combines mouse + keyboard into a single feature vector
        """

        if len(self.mouse_buffer) < 2:
            return None

        # mouse features
        velocities = []
        jitters = []

        for i in range(1, len(self.mouse_buffer)):
            (t1, x1, y1) = self.mouse_buffer[i - 1]
            (t2, x2, y2) = self.mouse_buffer[i]

            dt = max(t2 - t1, 1e-6)
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

            v = dist / dt
            velocities.append(v)

            if len(velocities) > 1:
                jitters.append(abs(velocities[-1] - velocities[-2]))

        # keyboard features
        dwells = [k[1] for k in self.key_buffer] if self.key_buffer else [0.2]

        feature_vector = [
            np.mean(velocities) if velocities else 0,
            np.std(velocities) if velocities else 0,
            np.mean(jitters) if jitters else 0,
            np.mean(dwells),
            len(self.key_buffer)
        ]

        return feature_vector

    # ===============================
    # MAIN SECURITY LOOP
    # ===============================
    def _evaluate_risk(self):

        features = self._extract_features()
        if features is None:
            return

        action, score = self.brain.get_action(features)

        # smooth risk accumulation (VERY IMPORTANT FIX)
        if action == 1:
            self.risk_score += (score / 100) * 2.5
        else:
            self.risk_score -= 0.5

        self.risk_score = np.clip(self.risk_score, 0, 100)

        # cleanup old data (time window)
        now = time.time()

        self.mouse_buffer = [
            m for m in self.mouse_buffer
            if now - m[0] <= self.window_size
        ]

        self.key_buffer = [
            k for k in self.key_buffer
            if now - k[0] <= self.window_size
        ]

        # final decision
        if self.risk_score > 70:
            self._handle_breach()

    # ===============================
    # MOUSE HANDLER
    # ===============================
    def _process_mouse(self, x, y):

        if not self.is_active:
            return

        now = time.time()

        self.mouse_buffer.append((now, x, y))

        # periodic evaluation
        if len(self.mouse_buffer) % 5 == 0:
            self._evaluate_risk()

    # ===============================
    # KEYBOARD HANDLERS
    # ===============================
    def _process_keys(self, key):
        self.key_press_times[key] = time.time()

    def _process_key_release(self, key):

        now = time.time()

        if key in self.key_press_times:
            dwell = now - self.key_press_times[key]

            self.key_buffer.append((now, dwell))

            del self.key_press_times[key]

            self._evaluate_risk()

    # ===============================
    # BREACH HANDLING
    # ===============================
    def _handle_breach(self):

        self.is_active = False

        print("Wontx AI: SECURITY BREACH DETECTED")

        if self.on_breach:
            self.on_breach()