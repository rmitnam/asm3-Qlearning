"""
Q-Learning Algorithm Implementation
Off-policy temporal difference learning.
"""

import numpy as np
import random
from collections import defaultdict
import config


class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy exploration.
    
    Q-Learning update rule:
    Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
    """
    
    def __init__(self, alpha=None, gamma=None, epsilon_start=None,
                 epsilon_end=None, epsilon_decay_episodes=None):
        """Initialize Q-Learning agent with linear epsilon decay."""
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
        """
        Get Q-value for a state-action pair.
        
        Args:
            state: Current state
            action: Action
            
        Returns:
            Q-value (0.0 if not seen before)
        """
        return self.q_table[(state, action)]
    
    def get_max_q_value(self, state):
        """
        Get the maximum Q-value for a state across all actions.
        
        Args:
            state: Current state
            
        Returns:
            Maximum Q-value
        """
        q_values = [self.get_q_value(state, a) for a in range(config.NUM_ACTIONS)]
        return max(q_values)
    
    def get_best_action(self, state):
        """
        Get the best action for a state (with random tie-breaking).
        
        Args:
            state: Current state
            
        Returns:
            Best action
        """
        # Get Q-values for all actions
        q_values = [self.get_q_value(state, a) for a in range(config.NUM_ACTIONS)]
        max_q = max(q_values)
        
        # Find all actions with maximum Q-value
        best_actions = [a for a in range(config.NUM_ACTIONS) if q_values[a] == max_q]
        
        # Random tie-breaking
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
            # Exploration: random action
            return random.randint(0, config.NUM_ACTIONS - 1)
        else:
            # Exploitation: best action
            return self.get_best_action(state)
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using Q-Learning update rule.
        
        Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        current_q = self.get_q_value(state, action)
        
        if done:
            # Terminal state: no future rewards
            target = reward
        else:
            # Non-terminal: consider future rewards
            max_next_q = self.get_max_q_value(next_state)
            target = reward + self.gamma * max_next_q
        
        # Q-Learning update
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[(state, action)] = new_q
        
        self.total_steps += 1
    
    def decay_epsilon(self):
        """Decay epsilon linearly from epsilon_start to epsilon_end over epsilon_decay_episodes."""
        self.episode_count += 1
        if self.episode_count >= self.epsilon_decay_episodes:
            self.epsilon = self.epsilon_end
        else:
            # Linear decay: epsilon = start - (start - end) * (episode / decay_episodes)
            decay_fraction = self.episode_count / self.epsilon_decay_episodes
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * decay_fraction
    
    def reset_epsilon(self):
        """Reset epsilon to initial value (for retraining)."""
        self.epsilon = self.epsilon_start
        self.episode_count = 0
    
    def get_policy(self, state):
        """
        Get the current policy for a state (action probabilities).
        
        Args:
            state: Current state
            
        Returns:
            Dictionary mapping actions to probabilities
        """
        best_action = self.get_best_action(state)
        policy = {}
        
        for a in range(config.NUM_ACTIONS):
            if a == best_action:
                # Best action gets (1 - epsilon) + epsilon/n_actions
                policy[a] = (1 - self.epsilon) + self.epsilon / config.NUM_ACTIONS
            else:
                # Other actions get epsilon/n_actions
                policy[a] = self.epsilon / config.NUM_ACTIONS
        
        return policy
    
    def save_q_table(self, filepath):
        """
        Save Q-table to file.
        
        Args:
            filepath: Path to save file
        """
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Q-table saved to {filepath}")
    
    def load_q_table(self, filepath):
        """
        Load Q-table from file.
        
        Args:
            filepath: Path to load file
        """
        import pickle
        with open(filepath, 'rb') as f:
            loaded_table = pickle.load(f)
            self.q_table = defaultdict(float, loaded_table)
        print(f"Q-table loaded from {filepath}")
    
    def get_stats(self):
        """
        Get training statistics.
        
        Returns:
            Dictionary with statistics
        """
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
        print(f"Q-Learning Agent Statistics:")
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Total Steps: {stats['total_steps']}")
        print(f"  Epsilon: {stats['epsilon']:.4f}")
        print(f"  Q-table Size: {stats['q_table_size']} state-action pairs")
        print(f"  Alpha: {stats['alpha']}")
        print(f"  Gamma: {stats['gamma']}")


if __name__ == "__main__":
    # Quick test of linear decay
    agent = QLearningAgent()
    print(f"Linear epsilon decay test:")
    print(f"  Start: {agent.epsilon_start}, End: {agent.epsilon_end}, Episodes: {agent.epsilon_decay_episodes}")
    for ep in [0, 1000, 2000, 3000, 4000]:
        agent.episode_count = 0
        agent.epsilon = agent.epsilon_start
        for _ in range(ep):
            agent.decay_epsilon()
        print(f"  Episode {ep}: epsilon = {agent.epsilon:.4f}")
