import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from arena_env import SpaceArenaEnv

# Create directories for models and logs
models_dir = "models"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def train(style_id, total_timesteps=100000):
    print(f"--- Starting Training for Control Style {style_id} ---")
    
    # Initialize environment
    # We pass the 'control_style' argument to the environment constructor
    env = make_vec_env(lambda: SpaceArenaEnv(control_style=style_id), n_envs=1)
    
    # Define the PPO model
    # We use a Multi-Layer Perceptron (MlpPolicy)
    # Hyperparameters tuned for stability
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log=log_dir,
        learning_rate=0.0003,
        gamma=0.99,
        batch_size=64,
        ent_coef=0.01 # Encourage exploration
    )

    # Train
    model.learn(total_timesteps=total_timesteps)
    
    # Save
    save_path = f"{models_dir}/ppo_space_arena_style_{style_id}"
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    env.close()

if __name__ == "__main__":
    # Train Style 1 (Thrust/Rotate)
    train(style_id=1, total_timesteps=100000)
    
    # Train Style 2 (Direct Directional)
    train(style_id=2, total_timesteps=100000)