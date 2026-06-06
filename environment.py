import numpy as np
import pandas as pd
import sqlite3
import os

class WontxAIEnv:
    """
    WontxAIEnv: A reinforcement learning environment designed to train the agent.
    This simulates user behavioral sessions, generating both owner patterns
    and adversarial simulations to harden the security agent.
    """
    def __init__(self, db_path='wontxai.db'):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Missing {db_path}. Please run collector.py first.")
        
        conn = sqlite3.connect(db_path)
        m_df = pd.read_sql_query("SELECT timestamp, x, y FROM movement_logs WHERE label='owner'", conn)
        k_df = pd.read_sql_query("SELECT dwell_time FROM keyboard_logs WHERE label='owner'", conn)
        conn.close()

        if m_df.empty: raise ValueError("The database is empty. Please collect data first.")

        # Data Processing for accurate telemetry
        m_df['dt'] = m_df['timestamp'].diff().fillna(0.01)
        m_df['dist'] = np.sqrt(m_df['x'].diff().fillna(0)**2 + m_df['y'].diff().fillna(0)**2)
        m_df['velocity'] = (m_df['dist'] / m_df['dt']).fillna(0)
        m_df['jitter'] = m_df['velocity'].diff().abs().fillna(0)
        
        if len(m_df) < 100 or len(k_df) < 100:
            raise ValueError("Insufficient owner data to build a reliable training baseline.")

        self.owner_jitters = m_df['jitter'].values
        self.owner_dwells = k_df['dwell_time'].values
        
        # Calculate baseline statistics for non-static normalization
        self.jitter_mean = np.mean(self.owner_jitters)
        self.jitter_std = np.std(self.owner_jitters) + 1e-6

        self.current_step = 0
        self.max_steps = 100
        self.is_real_owner = True

    def reset(self):
        self.current_step = 0
        return self._get_next_state()

    def _get_next_state(self):
        """
        Generates the next state based on owner patterns or simulated anomalies.
        """
        self.is_real_owner = np.random.choice([True, False], p=[0.7, 0.3])
        
        if self.is_real_owner:
            # Recreate realistic user behavior from historical jitter
            raw_val = np.random.choice(self.owner_jitters)
            z_score = abs(raw_val - self.jitter_mean) / self.jitter_std
            state = int(np.clip(z_score, 0, 10))
        else:
            # Simulate anomalous 'attacker' behavior (high variance/drift)
            state = np.random.randint(7, 11) 
            
        return state

    def step(self, state, action):
        """
        Implements a professional Reward/Penalty structure.
        Ensures the agent is optimized for security (alerting) without 
        failing on owner behavior (False Positives).
        """
        reward = 0
        done = False
        
        if self.is_real_owner:
            # Penalty for False Lockouts (Accuracy Focus)
            if action == 0: reward = 10 
            else: reward = -200 
        else:
            # Reward for catching Intruders (Security Focus)
            if action == 1: reward = 200
            else: reward = -300 # Heavy penalty for allowing breach
            
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            
        return self._get_next_state(), reward, done