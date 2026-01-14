"""
Utility functions for training and evaluation.
Includes plotting, statistics, and helper functions.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def smooth_curve(values, window=100):
    """
    Smooth a curve using moving average.
    
    Args:
        values: List of values
        window: Window size for moving average
        
    Returns:
        Smoothed values
    """
    if len(values) < window:
        return values
    
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')
    return smoothed


def plot_training_curves(rewards_history, steps_history=None, 
                         title="Training Progress", 
                         save_path=None, show=True):
    """
    Plot training curves (rewards and optionally steps per episode).
    
    Args:
        rewards_history: List of rewards per episode
        steps_history: Optional list of steps per episode
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to show the plot
    """
    fig, axes = plt.subplots(1, 2 if steps_history else 1, figsize=(15 if steps_history else 10, 5))
    
    if not steps_history:
        axes = [axes]
    
    episodes = range(len(rewards_history))
    
    # Plot rewards
    ax1 = axes[0]
    ax1.plot(episodes, rewards_history, alpha=0.3, color='blue', label='Raw')
    
    # Add smoothed curve
    if len(rewards_history) >= 100:
        smoothed = smooth_curve(rewards_history, window=100)
        smooth_episodes = range(99, len(rewards_history))
        ax1.plot(smooth_episodes, smoothed, color='red', linewidth=2, label='Smoothed (100 ep)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Rewards per Episode')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot steps if provided
    if steps_history:
        ax2 = axes[1]
        ax2.plot(episodes, steps_history, alpha=0.3, color='green', label='Raw')
        
        # Add smoothed curve
        if len(steps_history) >= 100:
            smoothed = smooth_curve(steps_history, window=100)
            smooth_episodes = range(99, len(steps_history))
            ax2.plot(smooth_episodes, smoothed, color='orange', linewidth=2, label='Smoothed (100 ep)')
        
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Steps')
        ax2.set_title('Steps per Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_comparison(results_dict, metric='reward', 
                   title="Algorithm Comparison",
                   save_path=None, show=True):
    """
    Plot comparison of multiple algorithms or configurations.
    
    Args:
        results_dict: Dictionary mapping algorithm names to reward histories
        metric: 'reward' or 'steps'
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to show the plot
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
    
    for i, (name, history) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        episodes = range(len(history))
        
        # Plot raw data with low alpha
        plt.plot(episodes, history, alpha=0.2, color=color)
        
        # Plot smoothed data
        if len(history) >= 100:
            smoothed = smooth_curve(history, window=100)
            smooth_episodes = range(99, len(history))
            plt.plot(smooth_episodes, smoothed, color=color, 
                    linewidth=2, label=name)
        else:
            plt.plot(episodes, history, color=color, linewidth=2, label=name)
    
    plt.xlabel('Episode')
    plt.ylabel(f'Total {metric.capitalize()}')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def compute_statistics(values, window=100):
    """
    Compute statistics for a list of values.
    
    Args:
        values: List of values
        window: Window for recent statistics
        
    Returns:
        Dictionary with statistics
    """
    values = np.array(values)
    recent_values = values[-window:] if len(values) >= window else values
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'recent_mean': np.mean(recent_values),
        'recent_std': np.std(recent_values),
        'total_episodes': len(values),
    }


def print_training_summary(rewards_history, steps_history, 
                          algorithm_name="Q-Learning", level_id=0):
    """
    Print a summary of training results.
    
    Args:
        rewards_history: List of rewards per episode
        steps_history: List of steps per episode
        algorithm_name: Name of the algorithm
        level_id: Level number
    """
    reward_stats = compute_statistics(rewards_history)
    steps_stats = compute_statistics(steps_history)
    
    print("\n" + "=" * 60)
    print(f"TRAINING SUMMARY: {algorithm_name} on Level {level_id}")
    print("=" * 60)
    
    print("\nReward Statistics:")
    print(f"  Overall Mean: {reward_stats['mean']:.2f} ± {reward_stats['std']:.2f}")
    print(f"  Recent Mean (last 100): {reward_stats['recent_mean']:.2f} ± {reward_stats['recent_std']:.2f}")
    print(f"  Min: {reward_stats['min']:.2f} | Max: {reward_stats['max']:.2f}")
    
    print("\nSteps Statistics:")
    print(f"  Overall Mean: {steps_stats['mean']:.1f} ± {steps_stats['std']:.1f}")
    print(f"  Recent Mean (last 100): {steps_stats['recent_mean']:.1f} ± {steps_stats['recent_std']:.1f}")
    print(f"  Min: {steps_stats['min']:.0f} | Max: {steps_stats['max']:.0f}")
    
    print(f"\nTotal Episodes: {len(rewards_history)}")
    print("=" * 60 + "\n")


def evaluate_policy(env, agent, num_episodes=10, render=False, verbose=True):
    """
    Evaluate a trained agent's policy.
    
    Args:
        env: Environment instance
        agent: Trained agent
        num_episodes: Number of evaluation episodes
        render: Whether to render episodes
        verbose: Whether to print episode results
        
    Returns:
        Dictionary with evaluation results
    """
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    if verbose:
        print(f"\nEvaluating policy for {num_episodes} episodes...")
        print("-" * 60)
    
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Select action (no exploration)
            action = agent.select_action(state, explore=False)
            state, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if render and hasattr(env, 'renderer'):
                env.renderer.render()
        
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        
        # Check success (positive reward means collected rewards)
        if total_reward > 0:
            success_count += 1
        
        if verbose:
            status = "SUCCESS" if total_reward > 0 else "FAILED"
            print(f"Episode {ep+1}: Reward = {total_reward:.2f}, Steps = {steps}, {status}")
    
    success_rate = success_count / num_episodes
    
    if verbose:
        print("-" * 60)
        print(f"Success Rate: {success_rate:.1%} ({success_count}/{num_episodes})")
        print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average Steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
        print()
    
    return {
        'rewards': episode_rewards,
        'steps': episode_steps,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_steps': np.mean(episode_steps),
        'std_steps': np.std(episode_steps),
        'success_rate': success_rate,
    }


if __name__ == "__main__":
    # Test plotting functions
    print("Testing utility functions\n")
    
    # Generate fake training data
    print("Generating sample training data...")
    episodes = 1000
    rewards = []
    steps = []
    
    for i in range(episodes):
        # Simulate improving performance
        base_reward = -10 + (i / episodes) * 13
        noise = np.random.randn() * 2
        rewards.append(base_reward + noise)
        
        base_steps = 100 - (i / episodes) * 50
        noise = np.random.randn() * 10
        steps.append(max(20, base_steps + noise))
    
    print(f"Generated {episodes} episodes of data\n")
    
    # Test statistics
    print("Computing statistics...")
    stats = compute_statistics(rewards)
    print(f"Reward stats: mean={stats['mean']:.2f}, recent_mean={stats['recent_mean']:.2f}")
    print()
    
    # Test plotting (don't show, just create and close)
    print("Testing plot generation...")
    plot_training_curves(rewards, steps, 
                        title="Test Training Curves",
                        show=False)
    print("✓ Single plot test passed\n")
    
    # Test comparison plot
    results = {
        'Algorithm A': rewards,
        'Algorithm B': [r + np.random.randn() for r in rewards]
    }
    plot_comparison(results, title="Test Comparison", show=False)
    print("✓ Comparison plot test passed\n")
    
    print("✓ All utility function tests passed!")
