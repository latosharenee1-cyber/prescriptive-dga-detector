# Filename: adaptive_playbook.py
import numpy as np

class AdaptivePlaybook:
    def __init__(self, actions, epsilon=0.1):
        self.actions = actions
        self.epsilon = epsilon  # Exploration rate
        self.q_values = {action: 0.0 for action in actions}
        self.action_counts = {action: 0 for action in actions}

    def choose_action(self):
        # With probability epsilon, explore a random action
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        # Otherwise, exploit the best-known action
        else:
            return max(self.q_values, key=self.q_values.get)

    def update_q_value(self, action, reward):
        self.action_counts[action] += 1
        # Update the average reward for the chosen action
        self.q_values[action] += (reward - self.q_values[action]) / self.action_counts[action]
