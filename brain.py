import numpy as np
import os

class WontxAIBrain:
    """
    Wontx AI Brain: The Neural Core.
    Handles decision making using a Q-Table mapping.
    """
    def __init__(self, states=11, actions=2):
        self.states = states
        self.actions = actions
        self.filename = 'q_table.npy'
        
        # Initialize or load Q-table
        if os.path.exists(self.filename):
            try:
                self.q_table = np.load(self.filename)
            except:
                self.q_table = np.zeros((states, actions))
        else:
            self.q_table = np.zeros((states, actions))

    def get_action(self, state):
        """
        Corrects the state handling. If 'state' is an int, we look it up in the Q-table.
        If your logic requires features, they must be derived or passed separately.
        """
        # 1. If state is passed as an int, use it directly for Q-table lookup
        if isinstance(state, int):
            # Epsilon-greedy action selection
            if np.random.rand() < 0.1:  # Exploration
                return np.random.randint(0, self.actions)
            else:  # Exploitation
                return np.argmax(self.q_table[state])
        
        # 2. If you absolutely need features for a calculation, ensure features is a list
        # For now, let's treat the state as the index directly:
        return np.argmax(self.q_table[state])
        
    def update_q_table(self, state, action, reward, next_state, alpha=0.2, gamma=0.95):
        current_q = self.q_table[state, action]
        max_future_q = np.max(self.q_table[next_state])
        
        # Bellman Equation
        self.q_table[state, action] = current_q + alpha * (reward + gamma * max_future_q - current_q)
        np.save(self.filename, self.q_table)

    def get_confidence_score(self, state):
        """
        Returns 0-100% confidence.
        """
        trust_val = self.q_table[state, 0]
        alert_val = self.q_table[state, 1]
        
        # Softmax normalization
        exp_t = np.exp(trust_val)
        exp_a = np.exp(alert_val)
        return (exp_t / (exp_t + exp_a)) * 100