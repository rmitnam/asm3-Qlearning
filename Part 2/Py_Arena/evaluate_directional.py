"""
Evaluation script for Arena RL Environment with Directional Controls
Loads a trained model and visualizes the agent playing the game
"""

import os
import sys
import argparse
import time

import pygame
import numpy as np
from stable_baselines3 import PPO, DQN

from arena_env_directional import ArenaEnvDirectional
from config import *


def find_best_model(model_dir: str) -> str:
    """
    Find the best model in a directory.
    Looks for 'best_model.zip' first, then 'final' models, then checkpoints.
    
    Args:
        model_dir: Directory to search for models
        
    Returns:
        Path to the best model found
    """
    # Priority order for model files
    priorities = [
        "best_model.zip",
        "directional_ppo_final.zip",
        "directional_dqn_final.zip",
    ]
    
    for filename in priorities:
        path = os.path.join(model_dir, filename)
        if os.path.exists(path):
            return path
            
    # Look for checkpoint files
    files = os.listdir(model_dir) if os.path.exists(model_dir) else []
    checkpoints = [f for f in files if f.endswith('.zip') and 'checkpoint' in f]
    if checkpoints:
        # Get the latest checkpoint
        checkpoints.sort()
        return os.path.join(model_dir, checkpoints[-1])
        
    return None


def evaluate_agent(
    model_path: str,
    algorithm: str = "ppo",
    n_episodes: int = 5,
    deterministic: bool = True,
    slow_mode: bool = False,
    human_play: bool = False
):
    """
    Evaluate a trained agent with visual rendering.
    
    Args:
        model_path: Path to the trained model
        algorithm: Algorithm used (ppo or dqn)
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        slow_mode: Add delay between steps for better visualization
        human_play: Allow human to play instead of agent
    """
    print(f"=" * 60)
    print(f"Evaluating Agent - Directional Controls")
    print(f"=" * 60)
    
    # Create environment with rendering
    env = ArenaEnvDirectional(render_mode="human")
    
    if not human_play:
        # Load the model
        print(f"Loading model from: {model_path}")
        if algorithm == "ppo":
            model = PPO.load(model_path, env=env)
        else:
            model = DQN.load(model_path, env=env)
        print("Model loaded successfully!")
    else:
        model = None
        print("Human play mode - Use keyboard controls")
        print("  W or UP: Move up")
        print("  S or DOWN: Move down")
        print("  A or LEFT: Move left")
        print("  D or RIGHT: Move right")
        print("  SPACE: Shoot (auto-aims at nearest enemy)")
        
    print(f"\nRunning {n_episodes} episodes...")
    print(f"Deterministic: {deterministic}")
    print(f"Slow mode: {slow_mode}")
    print("=" * 60)
    
    # Statistics
    episode_rewards = []
    episode_lengths = []
    episode_phases = []
    episode_kills = []
    episode_spawner_kills = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        
        while not done:
            if human_play:
                action = get_human_action_directional()
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            if slow_mode:
                time.sleep(0.02)
                
            # Handle pygame quit event
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    print("\nEvaluation terminated by user.")
                    return
                    
        # Record statistics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_phases.append(info.get('phase', 1))
        episode_kills.append(info.get('enemies_killed', 0))
        episode_spawner_kills.append(info.get('spawners_killed', 0))
        
        print(f"Episode {episode + 1} Results:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Phase Reached: {info.get('phase', 1)}")
        print(f"  Enemies Killed: {info.get('enemies_killed', 0)}")
        print(f"  Spawners Killed: {info.get('spawners_killed', 0)}")
        print(f"  Final Health: {info.get('player_health', 0)}")
        
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Episodes: {n_episodes}")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average Phase: {np.mean(episode_phases):.2f}")
    print(f"Average Enemies Killed: {np.mean(episode_kills):.1f}")
    print(f"Average Spawners Killed: {np.mean(episode_spawner_kills):.1f}")
    print(f"Best Reward: {max(episode_rewards):.2f}")
    print(f"Highest Phase: {max(episode_phases)}")
    print("=" * 60)
    
    env.close()


def get_human_action_directional() -> int:
    """
    Get action from keyboard input for directional controls.
    
    Returns:
        Action index (0-5)
    """
    keys = pygame.key.get_pressed()
    
    # Priority: Shoot > Movement
    if keys[pygame.K_SPACE]:
        return 5  # Shoot
    elif keys[pygame.K_w] or keys[pygame.K_UP]:
        return 1  # Move up
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        return 2  # Move down
    elif keys[pygame.K_a] or keys[pygame.K_LEFT]:
        return 3  # Move left
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        return 4  # Move right
    
    return 0  # No action


def list_available_models(base_dir: str = MODEL_DIR):
    """List all available trained models."""
    print("\nAvailable models:")
    print("-" * 40)
    
    if not os.path.exists(base_dir):
        print(f"  No models directory found at: {base_dir}")
        return
        
    found = False
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            # Check for directional models
            if 'directional' in item.lower():
                model_file = find_best_model(item_path)
                if model_file:
                    print(f"  {item}/")
                    print(f"    -> {os.path.basename(model_file)}")
                    found = True
                    
    if not found:
        print("  No directional models found.")
        print("  Run train_directional.py first to train a model.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent (Directional Controls)")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to trained model file or model directory"
    )
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="Algorithm used for training (default: ppo)"
    )
    parser.add_argument(
        "--episodes", "-e",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Add delay between steps for better visualization"
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Human play mode (keyboard controls)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available trained models"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
        
    if args.human:
        evaluate_agent(
            model_path=None,
            algorithm=args.algorithm,
            n_episodes=args.episodes,
            deterministic=True,
            slow_mode=args.slow,
            human_play=True
        )
        return
        
    # Find model path
    model_path = args.model
    
    if model_path is None:
        # Try to find a default model
        default_paths = [
            os.path.join(MODEL_DIR, "directional_ppo"),
            os.path.join(MODEL_DIR, "directional_dqn"),
        ]
        
        for path in default_paths:
            found = find_best_model(path)
            if found:
                model_path = found
                break
                
        if model_path is None:
            print("Error: No model specified and no default model found.")
            print("Use --model to specify a model path or --list to see available models.")
            print("Use --human to play the game manually.")
            return
            
    elif os.path.isdir(model_path):
        # Find best model in directory
        model_path = find_best_model(model_path)
        if model_path is None:
            print(f"Error: No model files found in {args.model}")
            return
            
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return
        
    evaluate_agent(
        model_path=model_path,
        algorithm=args.algorithm,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        slow_mode=args.slow,
        human_play=False
    )


if __name__ == "__main__":
    main()
