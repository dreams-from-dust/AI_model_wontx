import numpy as np
import os
import sqlite3

class WontxAIAgent:
    """
    Refactored WontxAIAgent: Uses dynamic calibration from the database
    to set behavior baselines, reducing false positives.
    """
    def __init__(self, states=11, actions=2, db_path='wontxai.db'):
        self.states = states
        self.actions = actions 
        self.filename = 'q_table.npy'
        
        # Dynamic Baselining: Calculate normal behavior from stored user data
        self.mouse_threshold, self.kb_dwell_std = self._calibrate(db_path)
        
        if os.path.exists(self.filename):
            self.q_table = np.load(self.filename)
        else:
            self.q_table = np.zeros((states, actions))

    def _calibrate(self, db_path):
        """Calculates personalized baselines from your history."""
        if not os.path.exists(db_path): return 500.0, 0.02
        
        try:
            conn = sqlite3.connect(db_path)
            # Fetch movement and dwell data
            m_data = np.array(conn.execute("SELECT x, y FROM movement_logs").fetchall())
            k_data = np.array(conn.execute("SELECT dwell_time FROM keyboard_logs").fetchall())
            conn.close()
            
            # Set threshold as Mean + 2 Standard Deviations (the 95% Confidence Interval)
            mouse_std = np.std(m_data) if len(m_data) > 10 else 500.0
            kb_std = np.std(k_data) if len(k_data) > 10 else 0.05
            return mouse_std * 2, kb_std * 2
        except:
            return 500.0, 0.05

    def get_decision(self, features, input_type='mouse'):
        """Maps features to state and returns (action, confidence)."""
        # Feature normalization based on dynamic calibration
        if input_type == 'mouse':
            norm_val = abs(features) / self.mouse_threshold
        else:
            norm_val = abs(features[0] - 0.05) / self.kb_dwell_std
            
        state = int(np.clip(norm_val * 10, 0, 10))
        
        q_vals = self.q_table[state]
        action = np.argmax(q_vals)
        
        # Confidence calculation
        exp_q = np.exp(q_vals - np.max(q_vals))
        confidence = (exp_q[1] / np.sum(exp_q)) * 100
        
        return action, confidence

    def update_brain(self, state, action, reward, next_state):
        alpha, gamma = 0.1, 0.9
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state])
        self.q_table[state, action] += alpha * (reward + gamma * max_future_q - current_q)
        np.save(self.filename, self.q_table)