# Q-Learning Gridworld Project

A reinforcement learning environment implementing Q-Learning and SARSA algorithms in a Pygame-based gridworld with a menu-driven interface.

## Features

- **Menu-Driven Interface**: Visual menu system with hover effects, level selection, and training progress display
- **7 Progressive Levels**: From basic pathfinding to complex navigation with monsters (Levels 0-6)
- **Two RL Algorithms**: Q-Learning (off-policy) and SARSA (on-policy)
- **Visual Rendering**: Pygame-based visualization with real-time feedback
- **Multiple Mechanics**: Apples (+1), keys, chests (+2), fire hazards, moving monsters (40% move chance)
- **Intrinsic Rewards**: Count-based exploration bonus using β/√(n(s)+1)

## Installation

```bash
pip install pygame numpy matplotlib
```

## Quick Start

```bash
python main.py
```

This launches the menu interface where you can:
- **Manual Play**: Control the agent with WASD/Arrow keys
- **Watch Trained Agent**: Automatically trains then demos the learned policy
- **Settings**: Toggle between Q-Learning/SARSA and enable intrinsic rewards

## Project Structure

```
├── main.py            # Menu interface (primary entry point)
├── config.py          # Configuration, parameters, and level definitions
├── gridworld.py       # Environment implementation
├── renderer.py        # Pygame visualization
├── q_learning.py      # Q-Learning algorithm
├── sarsa.py           # SARSA algorithm
├── train.py           # Command-line training script
└── demo_agent.py      # Simple demo script
```

## Level Descriptions

| Level | Name | Mechanics | Max Reward |
|-------|------|-----------|------------|
| 0 | Basic Pathfinding | Apples only | 3.0 |
| 1 | Fire Hazards | Apples + fire | 3.0 |
| 2 | Key and Chest | Apples + key + chest | 5.0 |
| 3 | Complex Navigation | Maze + key + chest | 5.0 |
| 4 | Monster Avoidance | Apples + monsters | 2.0 |
| 5 | Monsters + Key & Chest | All mechanics combined | 4.0 |
| 6 | Exploration Challenge | Maze with distant reward | 1.0 |

## Algorithms

### Q-Learning (Off-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```
- Uses greedy max for backup target
- Learns optimal policy regardless of exploration behavior
- Tends to find shorter paths near hazards

### SARSA (On-Policy)
```
Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]
```
- Uses actual next action for backup target
- Learns from actual behavior including exploration
- More conservative near hazards (accounts for exploration mistakes)

### Linear Epsilon Decay
```
ε(t) = ε_start - (ε_start - ε_end) × (t / T)
```

## State Representation

The state encoding varies by level complexity:

| Levels | State Format |
|--------|--------------|
| 0-1 | (x, y, collected_apples) |
| 2-3, 6 | (x, y, has_key, chest_opened, collected_apples) |
| 4-5 | (x, y, has_key, chest_opened, collected_apples, monster_direction) |

Monster direction uses (dx, dy, distance_category) instead of exact positions to prevent state-space explosion.

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| ALPHA | 0.1 | Learning rate |
| GAMMA | 0.95 | Discount factor |
| EPSILON_START | 1.0 | Initial exploration rate |
| EPSILON_END | 0.01 | Final exploration rate |
| EPSILON_DECAY_EPISODES | 4000 | Linear decay duration |
| MAX_EPISODES | 5000 | Training episodes |
| MAX_STEPS_PER_EPISODE | 200 | Step limit per episode |
| MONSTER_MOVE_PROBABILITY | 0.4 | Monster movement chance |
| INTRINSIC_REWARD_BETA | 0.1 | Exploration bonus strength |

## Controls

### Menu
- **Mouse**: Hover and click buttons
- **ESC**: Return to previous menu / quit

### In-Game
- **WASD / Arrow Keys**: Move agent (manual mode)
- **R**: Reset level
- **ESC**: Return to menu

## Evaluation Results

| Level | Algorithm | Success Rate | Avg Steps | Avg Reward |
|-------|-----------|--------------|-----------|------------|
| 0 | Q-Learning | 100% | 18 | 3.0 |
| 1 | Q-Learning | 100% | 18 | 3.0 |
| 1 | SARSA | 0% | 200 | 0.0 |
| 2 | Q-Learning | 100% | 25 | 5.0 |
| 3 | Q-Learning | 100% | 28 | 5.0 |
| 4 | Q-Learning | 90% | 39 | 1.8 |
| 5 | Q-Learning | 100% | 104 | 3.8 |
| 6 | Q-Learning | 100% | 18 | 1.0 |

Note: SARSA fails on Level 1 due to on-policy conservatism near fire hazards.
