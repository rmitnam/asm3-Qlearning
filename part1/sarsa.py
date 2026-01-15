"""
SARSA Algorithm Implementation
On-policy temporal difference learning.
"""

import numpy as np
import random
from collections import defaultdict
import config


class SARSAAgent:
    """
    SARSA agent with epsilon-greedy exploration.
    
    SARSA update rule (on-policy):
    Q(s, a) ← Q(s, a) + α[r + γ Q(s', a') - Q(s, a)]
    
    Key difference from Q-Learning: uses actual next action a' (from policy)
    rather than max Q-value.
    """
    
    def __init__(self, alpha=None, gamma=None, epsilon_start=None,
                 epsilon_end=None, epsilon_decay_episodes=None):
        """Initialize SARSA agent with linear epsilon decay."""
        self.alpha = alpha if alpha is not None else config.ALPHA
        self.gamma = gamma if gamma is not None else config.GAMMA
        self.epsilon_start = epsilon_start if epsilon_start is not None else config.EPSILON_START
        self.epsilon_end = epsilon_end if epsilon_end is not None else config.EPSILON_END
        self.epsilon_decay_episodes = epsilon_decay_episodes if epsilon_decay_episodes is not None else config.EPSILON_DECAY_EPISODES

        self.q_table = defaultdict(float)
        self.epsilon = self.epsilon_start
        self.episode_count = 0
        self.total_steps = 0
    
    def get_q_value(self, state, action):
        """Get Q-value for a state-action pair."""
        return self.q_table[(state, action)]
    
    def get_max_q_value(self, state):
        """Get the maximum Q-value for a state."""
        q_values = [self.get_q_value(state, a) for a in range(config.NUM_ACTIONS)]
        return max(q_values)
    
    def get_best_action(self, state):
        """Get the best action for a state (with random tie-breaking)."""
        q_values = [self.get_q_value(state, a) for a in range(config.NUM_ACTIONS)]
        max_q = max(q_values)
        best_actions = [a for a in range(config.NUM_ACTIONS) if q_values[a] == max_q]
        return random.choice(best_actions)
    
    def select_action(self, state, explore=True):
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            explore: If True, use epsilon-greedy; if False, always exploit
            
        Returns:
            Selected action
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, config.NUM_ACTIONS - 1)
        else:
            return self.get_best_action(state)
    
    def update(self, state, action, reward, next_state, next_action, done):
        """
        Update Q-value using SARSA update rule.
        
        Q(s, a) ← Q(s, a) + α[r + γ Q(s', a') - Q(s, a)]
        
        Key difference: uses Q(s', a') where a' is the ACTUAL next action
        taken by the policy, not max_a' Q(s', a').
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            next_action: Next action (actually selected by policy)
            done: Whether episode is done
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            target = reward
        else:
            # SARSA: use Q(s', a') where a' is actual next action
            next_q = self.get_q_value(next_state, next_action)
            target = reward + self.gamma * next_q
        
        # Update
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[(state, action)] = new_q
        
        self.total_steps += 1
    
    def decay_epsilon(self):
        """Decay epsilon linearly from epsilon_start to epsilon_end over epsilon_decay_episodes."""
        self.episode_count += 1
        if self.episode_count >= self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_end
        else:
            decay_fraction = self.episode_count / self.epsilon_decay_episodes
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_fraction
    
    def reset_epsilon(self):
        """Reset epsilon to initial value."""
        self.epsilon = self.epsilon_start
        self.episode_count = 0
    
    def save_q_table(self, filepath):
        """Save Q-table to file."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filepath}")
    
    def load_q_table(self, filepath):
        """Load Q-table from file."""
        import pickle
        with open(filepath, 'rb') as f:
            loaded_table = pickle.load(f)
            self.q_table = defaultdict(float, loaded_table)
        print(f"Q-table loaded from {filepath}")
    
    def get_stats(self):
        """Get training statistics."""
        return {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'q_table_size': len(self.q_table),
            'alpha': self.alpha,
            'gamma': self.gamma,
        }
    
    def print_stats(self):
        """Print training statistics."""
        stats = self.get_stats()
        print(f"SARSA Agent Statistics:")
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Total Steps: {stats['total_steps']}")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Q-table Size: {stats['q_table_size']} state-action pairs")
        print(f"  Alpha: {stats['alpha']}")
        print(f"  Gamma: {stats['gamma']}")


if __name__ == "__main__":
    # Quick test of linear decay
    agent = SARSAAgent()
    print(f"SARSA Linear epsilon decay test:")
    print(f"  Start: {agent.epsilon_start}, End: {agent.epsilon_end}, Episodes: {agent.epsilon_decay_episodes}")
