"""
Demo script to visualize trained agent.
Use this for video recording.
"""

import argparse
import time
from gridworld import GridWorld
from q_learning import QLearningAgent
from sarsa import SARSAAgent
from renderer import GridWorldRenderer
from train import train_agent
import config


def demo_trained_agent(level_id, algorithm='qlearning', num_episodes=3,
                       delay_ms=150, use_intrinsic=False, model_path=None):
    """Train and demo an agent on a level."""

    # Set intrinsic reward config
    config.USE_INTRINSIC_REWARD = use_intrinsic

    # Create fresh environment and agent
    env = GridWorld(level_id=level_id)

    if algorithm == 'qlearning':
        agent = QLearningAgent()
    else:
        agent = SARSAAgent()

    # Load model if provided, otherwise train
    if model_path:
        print(f"Loading model from {model_path}...")
        agent.load_q_table(model_path)
    else:
        print(f"Training {algorithm} on Level {level_id}...")
        if use_intrinsic:
            print("  (with intrinsic rewards)")
        train_agent(agent, env, max_episodes=5000, render=False, verbose=False)
        print("Training complete!")

    # Create renderer
    renderer = GridWorldRenderer(env, manual_control=False)

    print(f"\nRunning {num_episodes} demo episodes...")
    print("Press ESC or close window to stop.\n")

    for ep in range(num_episodes):
        # Create fresh env for each episode
        demo_env = GridWorld(level_id=level_id)
        renderer.env = demo_env  # Update renderer's env reference

        state = demo_env.reset()
        done = False
        total_reward = 0
        steps = 0

        print(f"Episode {ep + 1}:")

        while not done:
            # Get greedy action (no exploration)
            action = agent.select_action(state, explore=False)

            # Take step
            state, reward, done, info = demo_env.step(action)
            total_reward += reward
            steps += 1

            # Render
            if renderer.render(delay_ms=delay_ms) == False:
                print("Window closed.")
                renderer.close()
                return

        result = "SUCCESS" if total_reward > 0 else "FAILED"
        print(f"  {result}: {steps} steps, reward = {total_reward:.1f}")

        time.sleep(1)  # Pause between episodes

    print("\nDemo complete!")
    renderer.close()


def main():
    parser = argparse.ArgumentParser(description='Demo trained agent')
    parser.add_argument('--level', type=int, default=0, help='Level (0-6)')
    parser.add_argument('--algorithm', type=str, default='qlearning',
                       choices=['qlearning', 'sarsa'])
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--delay', type=int, default=150,
                       help='Delay between steps in ms')
    parser.add_argument('--intrinsic', action='store_true',
                       help='Use intrinsic rewards (Level 6)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to saved model')

    args = parser.parse_args()

    demo_trained_agent(
        level_id=args.level,
        algorithm=args.algorithm,
        num_episodes=args.episodes,
        delay_ms=args.delay,
        use_intrinsic=args.intrinsic,
        model_path=args.model
    )


if __name__ == "__main__":
    main()
