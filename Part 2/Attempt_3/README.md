# Arena RL - Deep Reinforcement Learning for Action Arena Games

A real-time Pygame arena game environment with deep reinforcement learning agents trained using Stable Baselines3. This project implements a complete RL pipeline including custom Gym environments, training scripts, and evaluation tools.

## 🎮 Game Overview

The arena is a simplified action game where:
- **Player Ship**: A controllable ship with health, movement, and shooting capabilities
- **Enemy Spawners**: Hexagonal structures that periodically spawn enemies
- **Enemies**: Red diamond-shaped entities that chase the player
- **Projectiles**: Player-fired bullets that damage enemies and spawners
- **Phase System**: Destroying all spawners progresses to harder phases

### Objective
Survive as long as possible while destroying enemy spawners to progress through phases. The game features continuous movement, collision detection, and a phase-based difficulty system.

## 📁 Project Structure

```
Attempt_3/
├── config.py                    # All configuration constants and hyperparameters
├── entities.py                  # Player, Enemy, Spawner, Projectile classes
├── arena_env_rotation.py        # Gym environment with rotation controls
├── arena_env_directional.py     # Gym environment with directional controls
├── train_rotation.py            # Training script for rotation agent
├── train_directional.py         # Training script for directional agent
├── evaluate_rotation.py         # Evaluation script for rotation agent
├── evaluate_directional.py      # Evaluation script for directional agent
├── logs/                        # TensorBoard logs (created during training)
├── models/                      # Saved models (created during training)
└── README.md                    # This file
```

## 🎯 Control Schemes

### Control Style 1: Rotation Movement
| Action | Effect |
|--------|--------|
| 0 | No action |
| 1 | Thrust forward |
| 2 | Rotate left |
| 3 | Rotate right |
| 4 | Shoot |

The ship has momentum and friction, requiring the player to plan ahead. This scheme is more challenging but offers greater control precision.

### Control Style 2: Directional Movement
| Action | Effect |
|--------|--------|
| 0 | No action |
| 1 | Move up |
| 2 | Move down |
| 3 | Move left |
| 4 | Move right |
| 5 | Shoot (auto-aims at nearest target) |

Direct 4-way movement with auto-aiming when shooting. Easier to learn but less precise control.

## 📊 Observation Space

### Rotation Controls (28 features)
| Index | Feature | Description |
|-------|---------|-------------|
| 0-1 | Player position | x, y normalized to [-1, 1] |
| 2-3 | Player velocity | vx, vy normalized |
| 4-5 | Player orientation | cos(angle), sin(angle) |
| 6 | Player health | Normalized [0, 1] |
| 7 | Current phase | Normalized [0, 1] |
| 8-11 | Nearest enemy | distance, angle, rel_x, rel_y |
| 12-15 | Second nearest enemy | distance, angle, rel_x, rel_y |
| 16-19 | Nearest spawner | distance, angle, rel_x, rel_y |
| 20-23 | Second nearest spawner | distance, angle, rel_x, rel_y |
| 24 | Enemy count | Normalized count |
| 25 | Spawner count | Normalized count |
| 26 | Can shoot | 1.0 if cooldown ready |
| 27 | Time remaining | Normalized remaining steps |

### Directional Controls (26 features)
Same structure but without orientation features (indices 4-5), as direction is determined by movement.

## 🏆 Reward Structure

| Event | Reward | Justification |
|-------|--------|---------------|
| Kill enemy | +10.0 | Encourages combat engagement |
| Kill spawner | +50.0 | Primary objective - eliminates threat source |
| Progress to next phase | +100.0 | Major milestone reward |
| Take damage | -5.0 | Discourages reckless behavior |
| Death | -200.0 | Strong penalty to encourage survival |
| Survival bonus | +0.1/step | Small reward for staying alive |
| Step penalty | -0.001/step | Encourages efficiency |
| Projectile hit | +2.0 | Rewards accurate shooting |

### Shaping Rewards (Justified)
- **Approach spawner bonus**: Up to +0.02/step when close to spawners
  - *Justification*: Guides early exploration toward main objectives without overshadowing task rewards

## 🚀 Installation

### Requirements
```bash
pip install pygame numpy gymnasium stable-baselines3[extra] tensorboard
```

### Verify Installation
```python
python arena_env_rotation.py  # Test rotation environment
python arena_env_directional.py  # Test directional environment
```

## 🎓 Training

### Train Rotation Agent (PPO)
```bash
# Default training (500k timesteps)
python train_rotation.py

# Custom training
python train_rotation.py --timesteps 300000 --n-envs 8 --name my_experiment
```

### Train Directional Agent (PPO)
```bash
# Default training
python train_directional.py

# Custom training with different learning rate
python train_directional.py -t 400000 -lr 0.0001
```

### Training Arguments
| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| --algorithm | -a | ppo | RL algorithm (ppo/dqn) |
| --timesteps | -t | 500000 | Total training timesteps |
| --n-envs | -n | 4 | Parallel environments (PPO) |
| --learning-rate | -lr | 3e-4 | Learning rate |
| --seed | -s | 42 | Random seed |
| --name | -N | auto | Experiment name |

### Monitor Training
```bash
tensorboard --logdir=logs
```
Open http://localhost:6006 in your browser.

## 📈 Evaluation

### Evaluate Trained Agent
```bash
# Rotation agent
python evaluate_rotation.py --model models/rotation_ppo

# Directional agent  
python evaluate_directional.py --model models/directional_ppo
```

### Human Play Mode
```bash
# Play rotation controls manually
python evaluate_rotation.py --human

# Play directional controls manually
python evaluate_directional.py --human
```

### Evaluation Arguments
| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| --model | -m | auto | Model path or directory |
| --algorithm | -a | ppo | Algorithm used |
| --episodes | -e | 5 | Episodes to run |
| --stochastic | - | False | Use stochastic actions |
| --slow | - | False | Slow visualization |
| --human | - | False | Human play mode |
| --list | - | False | List available models |

## 🧠 Neural Network Architecture

Both agents use the same MLP architecture:
- **Policy Network**: 128 → 128 neurons (2 hidden layers)
- **Value Network**: 128 → 128 neurons (2 hidden layers)
- **Activation**: ReLU
- **Output**: Softmax over discrete actions

### PPO Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 3e-4 | Optimizer learning rate |
| N Steps | 2048 | Steps per update |
| Batch Size | 64 | Mini-batch size |
| Epochs | 10 | Passes over data |
| Gamma | 0.99 | Discount factor |
| GAE Lambda | 0.95 | GAE parameter |
| Clip Range | 0.2 | PPO clip range |
| Entropy Coef | 0.01 | Exploration bonus |

## 🎨 Visualization

### Game Elements
- **Player Ship**: Cyan triangle pointing in movement direction
- **Enemies**: Red diamonds that move toward player
- **Spawners**: Pulsing purple hexagons with spawn timer indicator
- **Projectiles**: Yellow glowing circles
- **Effects**: Particle explosions on kills and damage

### HUD Information
- Phase counter
- Health bar
- Enemy count
- Spawner count
- Kill statistics
- Reward tracker
- Step counter

## 📋 Episode Termination

An episode ends when:
1. **Player Death**: Health reaches 0
2. **Max Steps**: 3000 steps (configurable)
3. **Victory**: All spawners destroyed in final phase (Phase 5)

## 🔧 Configuration

All game parameters can be adjusted in `config.py`:

### Window Settings
```python
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 680
FPS = 60
```

### Player Settings
```python
PLAYER_MAX_HEALTH = 100
PLAYER_MAX_SPEED = 5.0
PLAYER_SHOOT_COOLDOWN = 10
```

### Difficulty Settings
```python
INITIAL_SPAWNERS = 2
SPAWNERS_PER_PHASE = 1
MAX_PHASE = 5
```

## 📊 Expected Performance

After ~500k timesteps of training:
- **Phase Reached**: Consistently reach Phase 2-3
- **Enemies Killed**: 20-50 per episode
- **Spawners Killed**: 2-5 per episode
- **Survival Time**: 1000-2000 steps

Performance varies based on:
- Random spawner placement
- Enemy behavior randomness
- Action exploration during training

## 🐛 Troubleshooting

### Common Issues

1. **Pygame not displaying**: Ensure display is available (not running headless)
2. **CUDA errors**: Set `device='cpu'` in model creation if no GPU
3. **Memory issues**: Reduce `n_envs` or `buffer_size`

### Tips
- Start with shorter training runs to test configuration
- Use TensorBoard to monitor learning progress
- Adjust rewards if agent gets stuck in local optima

## 📚 References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium (OpenAI Gym) Documentation](https://gymnasium.farama.org/)
- [Pygame Documentation](https://www.pygame.org/docs/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## 📝 License

This project is for educational purposes as part of a Game AI course assignment.
