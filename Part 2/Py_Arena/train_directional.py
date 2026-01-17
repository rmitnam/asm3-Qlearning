"""
Training script for Arena RL Environment with Directional Controls
Uses PPO algorithm from Stable Baselines3 with TensorBoard logging
"""

import os
import sys
import argparse
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback,
    CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from arena_env_directional import ArenaEnvDirectional
from config import *


def make_env(rank: int, seed: int = 0):
    """
    Create a wrapped environment for vectorized training.
    
    Args:
        rank: Index of the environment
        seed: Random seed
        
    Returns:
        Function that creates the environment
    """
    def _init():
        env = ArenaEnvDirectional(render_mode=None)  # No rendering during training
        env.reset(seed=seed + rank)
        return Monitor(env)
    set_random_seed(seed)
    return _init


def train_ppo(
    total_timesteps: int = TOTAL_TIMESTEPS,
    n_envs: int = 4,
    learning_rate: float = PPO_LEARNING_RATE,
    seed: int = 42,
    experiment_name: str = None
):
    """
    Train a PPO agent on the directional-controlled arena environment.
    
    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for the optimizer
        seed: Random seed for reproducibility
        experiment_name: Name for the experiment (for logging)
    """
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Use simple default name for easy evaluation
    if experiment_name is None:
        experiment_name = "directional_ppo"
    
    log_path = os.path.join(LOG_DIR, experiment_name)
    model_path = os.path.join(MODEL_DIR, experiment_name)
    
    print(f"=" * 60)
    print(f"Training PPO Agent - Directional Controls")
    print(f"=" * 60)
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Parallel Environments: {n_envs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Seed: {seed}")
    print(f"Log Path: {log_path}")
    print(f"Model Path: {model_path}")
    print(f"=" * 60)
    
    # Create vectorized environment
    print("\nCreating training environments...")
    env = DummyVecEnv([make_env(i, seed) for i in range(n_envs)])
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(0, seed + 100)])
    
    # Create PPO model with custom network architecture
    print("\nInitializing PPO model...")
    policy_kwargs = dict(
        net_arch=dict(
            pi=POLICY_NET_ARCH,  # Policy network
            vf=POLICY_NET_ARCH   # Value network
        )
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE,
        n_epochs=PPO_N_EPOCHS,
        gamma=PPO_GAMMA,
        gae_lambda=PPO_GAE_LAMBDA,
        clip_range=PPO_CLIP_RANGE,
        ent_coef=PPO_ENT_COEF,
        vf_coef=PPO_VF_COEF,
        max_grad_norm=PPO_MAX_GRAD_NORM,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=log_path
    )
    
    print(f"\nModel Architecture:")
    print(f"  Policy Network: {POLICY_NET_ARCH}")
    print(f"  Value Network: {POLICY_NET_ARCH}")
    print(f"  Total Parameters: {sum(p.numel() for p in model.policy.parameters()):,}")
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=EVAL_FREQ // n_envs,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // n_envs,
        save_path=model_path,
        name_prefix="directional_ppo_checkpoint"
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    print(f"\nMonitor training progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_path}")
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = os.path.join(model_path, "directional_ppo_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, model_path


def train_dqn(
    total_timesteps: int = TOTAL_TIMESTEPS,
    learning_rate: float = DQN_LEARNING_RATE,
    seed: int = 42,
    experiment_name: str = None
):
    """
    Train a DQN agent on the directional-controlled arena environment.
    
    Note: DQN is included as an alternative. PPO typically performs better
    for this type of environment, but DQN can be useful for comparison.
    
    Args:
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for the optimizer
        seed: Random seed for reproducibility
        experiment_name: Name for the experiment (for logging)
    """
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Use simple default name for easy evaluation
    if experiment_name is None:
        experiment_name = "directional_dqn"
    
    log_path = os.path.join(LOG_DIR, experiment_name)
    model_path = os.path.join(MODEL_DIR, experiment_name)
    
    print(f"=" * 60)
    print(f"Training DQN Agent - Directional Controls")
    print(f"=" * 60)
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Seed: {seed}")
    print(f"Log Path: {log_path}")
    print(f"Model Path: {model_path}")
    print(f"=" * 60)
    
    # Create environment (DQN doesn't support vectorized envs out of the box)
    print("\nCreating training environment...")
    env = Monitor(ArenaEnvDirectional(render_mode=None))
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = Monitor(ArenaEnvDirectional(render_mode=None))
    
    # Create DQN model
    print("\nInitializing DQN model...")
    policy_kwargs = dict(
        net_arch=POLICY_NET_ARCH
    )
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=DQN_BUFFER_SIZE,
        learning_starts=DQN_LEARNING_STARTS,
        batch_size=DQN_BATCH_SIZE,
        tau=DQN_TAU,
        gamma=DQN_GAMMA,
        train_freq=DQN_TRAIN_FREQ,
        target_update_interval=DQN_TARGET_UPDATE_INTERVAL,
        exploration_fraction=DQN_EXPLORATION_FRACTION,
        exploration_final_eps=DQN_EXPLORATION_FINAL_EPS,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=log_path
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_path,
        log_path=log_path,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_path,
        name_prefix="directional_dqn_checkpoint"
    )
    
    callbacks = CallbackList([eval_callback, checkpoint_callback])
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)
    print(f"\nMonitor training progress with TensorBoard:")
    print(f"  tensorboard --logdir={log_path}")
    print()
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = os.path.join(model_path, "directional_dqn_final")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, model_path


def main():
    parser = argparse.ArgumentParser(description="Train RL agent for Arena (Directional Controls)")
    parser.add_argument(
        "--algorithm", "-a",
        type=str,
        default="ppo",
        choices=["ppo", "dqn"],
        help="RL algorithm to use (default: ppo)"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS})"
    )
    parser.add_argument(
        "--n-envs", "-n",
        type=int,
        default=4,
        help="Number of parallel environments for PPO (default: 4)"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=None,
        help="Learning rate (default: algorithm-specific)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--name", "-N",
        type=str,
        default=None,
        help="Experiment name for logging"
    )
    
    args = parser.parse_args()
    
    if args.algorithm == "ppo":
        lr = args.learning_rate if args.learning_rate else PPO_LEARNING_RATE
        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=lr,
            seed=args.seed,
            experiment_name=args.name
        )
    else:
        lr = args.learning_rate if args.learning_rate else DQN_LEARNING_RATE
        train_dqn(
            total_timesteps=args.timesteps,
            learning_rate=lr,
            seed=args.seed,
            experiment_name=args.name
        )


if __name__ == "__main__":
    main()
