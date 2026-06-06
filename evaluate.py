import numpy as np
import sqlite3
import os
from agent import WontxAIAgent

def run_evaluation():
    """
    Evaluates the agent using a smoothed temporal window to filter 
    out jitter-based noise and improve accuracy measurement.
    """
    # Safety Check
    if not os.path.exists('wontxai.db'):
        print("Error: wontxai.db not found. Please run collector.py.")
        return

    agent = WontxAIAgent()
    conn = sqlite3.connect('wontxai.db')
    cursor = conn.cursor()
    
    # Configuration
    WINDOW_SIZE = 5  # Smoothing window
    
    # Fetch test data
    cursor.execute("SELECT x, y, label FROM movement_logs ORDER BY id DESC LIMIT 500")
    mouse_data = cursor.fetchall()
    cursor.execute("SELECT dwell_time, flight_time, label FROM keyboard_logs ORDER BY id DESC LIMIT 500")
    kb_data = cursor.fetchall()
    conn.close()

    def evaluate_stream(data, input_type):
        """Processes data in sliding windows to emulate real-world monitoring."""
        if not data: return 0, 0
        correct_windows = 0
        total_windows = 0
        
        # Process in chunks of WINDOW_SIZE
        for i in range(0, len(data) - WINDOW_SIZE, WINDOW_SIZE):
            batch = data[i:i + WINDOW_SIZE]
            decisions = []
            
            for item in batch:
                if input_type == 'mouse':
                    # item = (x, y, label)
                    action, _ = agent.get_decision(abs(item[0]), input_type='mouse')
                else:
                    # item = (dwell, flight, label)
                    action, _ = agent.get_decision((item[0], item[1]), input_type='keyboard')
                decisions.append(action)
            
            # Majority vote for the window
            final_decision = 1 if sum(decisions) > (WINDOW_SIZE / 2) else 0
            # Check against labels (assuming consistent labels in batch)
            expected = 0 if batch[0][-1] == 'owner' else 1
            
            if final_decision == expected:
                correct_windows += 1
            total_windows += 1
            
        return correct_windows, total_windows

    # Calculate results
    m_correct, m_total = evaluate_stream(mouse_data, 'mouse')
    k_correct, k_total = evaluate_stream(kb_data, 'keyboard')
    
    total_correct = m_correct + k_correct
    total_tests = m_total + k_total
    
    if total_tests > 0:
        accuracy = (total_correct / total_tests) * 100
        print(f"--- WONTX AI Evaluation ---")
        print(f"Processed {total_tests} behavioral windows.")
        print(f"Combined System Accuracy: {accuracy:.2f}%")
        print(f"Status: {'High Precision' if accuracy > 80 else 'Calibration Required'}")
    else:
        print("Error: Insufficient data for evaluation.")

if __name__ == "__main__":
    run_evaluation()