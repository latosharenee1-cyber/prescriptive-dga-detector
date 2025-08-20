# Filename: run_simulation.py
from adaptive_playbook import AdaptivePlaybook

actions = ["block_sender", "scan_attachment", "isolate_host"]
playbook = AdaptivePlaybook(actions)

# Simulate 5000 alerts. Let's assume for this scenario, "scan_attachment" is the optimal action.
for i in range(5000):
    chosen_action = playbook.choose_action()
    reward = 1.0 if chosen_action == "scan_attachment" else -1.0
    playbook.update_q_value(chosen_action, reward)

# Print the final learned values for each action
print("Final learned Q-values:")
print(playbook.q_values)
