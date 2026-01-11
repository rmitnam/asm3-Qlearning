import time
from stable_baselines3 import PPO
from arena_env import SpaceArenaEnv

def evaluate(style_id):
    model_path = f"models/ppo_space_arena_style_{style_id}.zip"
    
    print(f"Loading model: {model_path}")
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print("Model not found! Please run train.py first.")
        return

    # Create environment with render_mode='human' for visualization
    env = SpaceArenaEnv(control_style=style_id, render_mode="human")
    
    obs, _ = env.reset()
    done = False
    truncated = False
    
    print("Starting simulation... Press Ctrl+C to stop.")
    try:
        while not done and not truncated:
            # Predict action
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            
            # Optional: Slow down if it's too fast, though pygame clock handles FPS
            # time.sleep(0.01) 
            
            if done or truncated:
                print("Episode finished. Resetting...")
                obs, _ = env.reset()
                done = False
                truncated = False
                
    except KeyboardInterrupt:
        print("Evaluation stopped.")
    finally:
        env.close()

if __name__ == "__main__":
    # Ask user which style to watch
    choice = input("Enter control style to watch (1 for Rotate/Thrust, 2 for Direct): ")
    if choice in ['1', '2']:
        evaluate(int(choice))
    else:
        print("Invalid choice.")