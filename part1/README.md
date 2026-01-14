# Q-Learning Gridworld Project

A reinforcement learning environment implementing Q-Learning and SARSA algorithms in a Pygame-based gridworld.

## Features

- **6 Progressive Levels**: From basic pathfinding to complex navigation with monsters
- **Two RL Algorithms**: Q-Learning (off-policy) and SARSA (on-policy)
- **Visual Rendering**: Pygame-based visualization
- **Multiple Mechanics**: Apples, keys, chests, fire hazards, moving monsters
- **Intrinsic Rewards**: Exploration bonus support

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training

```bash
# Basic training
python train.py --level 0 --algorithm qlearning

# With visualization
python train.py --level 0 --algorithm qlearning --render

# With evaluation
python train.py --level 0 --algorithm qlearning --eval --eval-episodes 10

# Save results
python train.py --level 0 --algorithm qlearning \
    --save-plot results/training.png \
    --save-agent models/agent.pkl
```

### Manual Testing

```bash
# Play level manually (use arrow keys or WASD)
python renderer.py 0
```

## Project Structure

```
├── config.py          # Configuration and level definitions
├── gridworld.py       # Environment implementation
├── renderer.py        # Pygame visualization
├── q_learning.py      # Q-Learning algorithm
├── sarsa.py           # SARSA algorithm
├── train.py           # Training script
├── utils.py           # Utilities (plotting, evaluation)
└── requirements.txt   # Dependencies
```

## Level Descriptions

- **Level 0**: Basic pathfinding to apples
- **Level 1**: Navigate around fire hazards
- **Level 2**: Collect key to open chest
- **Level 3**: Complex maze with multiple rewards
- **Level 4**: Avoid moving monsters
- **Level 5**: Monsters + key-chest mechanics
- **Level 6**: Exploration challenge with intrinsic rewards

## Algorithms

### Q-Learning (Off-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```
- Learns optimal policy while exploring
- More aggressive near hazards

### SARSA (On-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```
- Learns from actual behavior
- More conservative approach

## Configuration

Edit `config.py` to adjust:
- Learning rate (ALPHA)
- Discount factor (GAMMA)
- Exploration parameters (EPSILON_START, EPSILON_END, EPSILON_DECAY)
- Maximum episodes and steps
- Rendering options

## Results

| Level | Q-Learning | SARSA |
|-------|------------|-------|
| 0 - Basic | 100% | 100% |
| 1 - Fire | 100% | 0% |
| 2 - Key-Chest | 100% | 100% |
| 3 - Complex | 100% | 100% |
| 4 - Monsters | 10% | 0% |

Q-Learning achieves optimal 18-step path on Level 0.

## License

Educational project for reinforcement learning coursework.
