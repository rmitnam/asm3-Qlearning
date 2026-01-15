"""
Training script for Q-Learning and SARSA agents.
Handles training loop, evaluation, and visualization.
"""

import argparse
import time
from pathlib import Path

import config
from gridworld import GridWorld
from q_learning import QLearningAgent
from renderer import GridWorldRenderer
from utils import (plot_training_curves, print_training_summary, 
                   evaluate_policy, plot_comparison)


def train_agent(agent, env, max_episodes=None, render=False, verbose=True):
    """
    Train an agent on an environment.
    
    Args:
        agent: QLearningAgent or SARSAAgent instance
        env: GridWorld environment
        max_episodes: Maximum number of training episodes
        render: Whether to render during training
        verbose: Whether to print progress
        
    Returns:
        (rewards_history, steps_history)
    """
    max_episodes = max_episodes or config.MAX_EPISODES
    
    rewards_history = []
    steps_history = []
    
    # Create renderer if needed
    renderer = None
    if render:
        renderer = GridWorldRenderer(env, manual_control=False)
    
    if verbose:
        print(f"\nStarting training for {max_episodes} episodes...")
        print(f"Level: {env.level_id} - {env.level_data['name']}")
        print(f"Agent: {agent.__class__.__name__}")
        print("-" * 60)
    
    start_time = time.time()
    
    for episode in range(max_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0

        # For SARSA: select first action
        action = agent.select_action(state, explore=True)

        while not done:
            # Take step
            next_state, reward, done, info = env.step(action)

            # Add intrinsic reward for updates only (env reward unchanged)
            update_reward = reward
            if config.USE_INTRINSIC_REWARD:
                update_reward += env.get_intrinsic_reward(next_state)

            # Select next action (needed for SARSA)
            next_action = agent.select_action(next_state, explore=True) if not done else None

            # Update agent (works for both Q-Learning and SARSA)
            if hasattr(agent, 'update') and 'next_action' in agent.update.__code__.co_varnames:
                # SARSA: needs next_action
                agent.update(state, action, update_reward, next_state, next_action, done)
            else:
                # Q-Learning: doesn't need next_action
                agent.update(state, action, update_reward, next_state, done)

            # Render if enabled
            if render and (config.RENDER_DURING_TRAINING or
                          (episode % config.RENDER_FREQUENCY == 0)):
                if renderer.render(delay_ms=50) == False:
                    # User closed window
                    if renderer:
                        renderer.close()
                    return rewards_history, steps_history

            state = next_state
            action = next_action
            episode_reward += reward
            episode_steps += 1
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record statistics
        rewards_history.append(episode_reward)
        steps_history.append(episode_steps)
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            recent_reward = sum(rewards_history[-100:]) / 100
            recent_steps = sum(steps_history[-100:]) / 100
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            
            print(f"Episode {episode+1}/{max_episodes} | "
                  f"Reward: {recent_reward:.2f} | "
                  f"Steps: {recent_steps:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Speed: {eps_per_sec:.1f} ep/s")
    
    if renderer:
        renderer.close()
    
    if verbose:
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"Training complete! Total time: {elapsed:.1f}s")
        print(f"Average speed: {max_episodes/elapsed:.1f} episodes/second")
    
    return rewards_history, steps_history


def main():
    """Main training function with command-line interface."""
    parser = argparse.ArgumentParser(description='Train RL agents on GridWorld')
    parser.add_argument('--level', type=int, default=0,
                       help='Level to train on (0-6)')
    parser.add_argument('--algorithm', type=str, default='qlearning',
                       choices=['qlearning', 'sarsa'],
                       help='Algorithm to use')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes (default from config)')
    parser.add_argument('--render', action='store_true',
                       help='Render during training')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation after training')
    parser.add_argument('--eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Path to save training plot')
    parser.add_argument('--save-agent', type=str, default=None,
                       help='Path to save trained agent Q-table')
    parser.add_argument('--intrinsic', action='store_true',
                       help='Enable intrinsic rewards (for Level 6)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Set intrinsic reward config
    if args.intrinsic:
        config.USE_INTRINSIC_REWARD = True
        print("Intrinsic rewards enabled")
    
    # Create environment
    env = GridWorld(level_id=args.level)
    
    # Create agent
    if args.algorithm == 'qlearning':
        agent = QLearningAgent()
    else:
        # SARSA will be implemented in Phase 5
        from sarsa import SARSAAgent
        agent = SARSAAgent()
    
    # Train
    rewards, steps = train_agent(
        agent, env,
        max_episodes=args.episodes,
        render=args.render,
        verbose=not args.quiet
    )
    
    # Print summary
    if not args.quiet:
        print_training_summary(rewards, steps, 
                              algorithm_name=agent.__class__.__name__,
                              level_id=args.level)
    
    # Plot training curves
    plot_title = f"{agent.__class__.__name__} on Level {args.level}"
    if args.intrinsic:
        plot_title += " (with intrinsic rewards)"
    
    plot_training_curves(
        rewards, steps,
        title=plot_title,
        save_path=args.save_plot,
        show=not args.quiet
    )
    
    # Save agent if requested
    if args.save_agent:
        Path(args.save_agent).parent.mkdir(parents=True, exist_ok=True)
        agent.save_q_table(args.save_agent)
    
    # Evaluation
    if args.eval:
        if args.render:
            # Create new renderer for evaluation
            renderer = GridWorldRenderer(env, manual_control=False)
            
        eval_results = evaluate_policy(
            env, agent,
            num_episodes=args.eval_episodes,
            render=False,  # We'll handle rendering manually
            verbose=not args.quiet
        )
        
        # Render a few successful episodes
        if args.render and not args.quiet:
            print("\nRendering evaluation episodes...")
            for i in range(min(3, args.eval_episodes)):
                state = env.reset()
                done = False
                
                while not done:
                    action = agent.select_action(state, explore=False)
                    state, reward, done, info = env.step(action)
                    
                    if renderer.render(delay_ms=100) == False:
                        break
                
                time.sleep(1)  # Pause between episodes
            
            renderer.close()
    
    print("\nTraining session complete!")


if __name__ == "__main__":
    main()
