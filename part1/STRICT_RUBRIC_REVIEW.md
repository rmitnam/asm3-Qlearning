# Q-Learning Gridworld Project — Strict Rubric Review

**Date:** 2026-01-15  
**Scope:** `part1/`

## 1) Executive summary (strict)

Overall: solid foundation. Pygame gridworld + Q-learning + SARSA are present and mostly correct.

**Two major rubric risks (likely mark deductions):**

1) **Epsilon decay is multiplicative, not linear.**
   - Requirement explicitly says “linear epsilon decay from epsilonStart to epsilonEnd (config-driven)”.
   - Current implementation uses `epsilon = max(end, epsilon * decay)`.

2) **Intrinsic reward is not implemented exactly as specified.**
   - **Formula mismatch:** spec uses $\beta / \sqrt{n(s)+1}$; current uses $\beta / \sqrt{n(s)}$.
   - **Spec requires env rewards unchanged** and intrinsic reward added for **Q/SARSA updates**; current code adds intrinsic inside `env.step()` and changes returned reward stream.

**Additional strict-marker risks:**
- Death reward is set to `-10.0`. Spec doesn’t explicitly allow/require a negative death reward.
- If an item is collected and then a monster moves into the agent in the same step, the reward is overwritten to death reward (item reward is lost). Might be judged incorrect.
- README claims SARSA is 0% on Level 1, undermining “demonstrate SARSA differs (more conservative)” unless justified with curves and explanation.

## 2) Project inventory

Key files:
- `config.py`: training parameters + level layouts + constants
- `gridworld.py`: environment dynamics, rewards, termination, monster movement, intrinsic tracking
- `renderer.py`: Pygame rendering + manual control
- `q_learning.py`: Q-learning implementation
- `sarsa.py`: SARSA implementation
- `train.py`: training loop + CLI
- `utils.py`: plotting + evaluation utilities

Entry points:
- `python train.py --level N --algorithm qlearning|sarsa [--render] [--eval] [--intrinsic]`
- `python renderer.py N` (manual play)

## 3) Requirements checklist (strict)

### A) Gridworld rules (environment)

**A1. Must be Pygame-rendered, animated, interactive (not console/text).**  
**Status:** PASS

**A2. Mechanics match spec:**
- Moves: up/down/left/right — **PASS**
- Rocks block movement; invalid move => no movement — **PASS**
- Fire or monsters cause immediate death — **PASS**
- Apples give +1 — **PASS**
- Keys give 0 reward but unlock chests — **PASS**
- Opening chest gives +2 — **PASS**
- Episode ends when all collectible rewards obtained or agent dies — **PASS**
- Monsters: 40% chance to move after each agent action — **PASS**
- Multiple levels with different layouts — **PASS**

**Strict caveats:**
- Death reward `-10.0` is not specified by the assignment text.
- Reward overwrite on post-move monster collision can drop collected item reward.
- Monsters can move onto reward tiles; may reduce learnability or change expected dynamics.

### B) Task 1 — Q-learning Level 0

- B1 Epsilon-greedy selection — **PASS**
- B2 Correct Q-learning update (off-policy max next state) — **PASS**
- B3 **Linear** epsilon decay (config-driven) — **FAIL (strict)** (currently multiplicative)
- B4 Random tie-breaking — **PASS**
- B5 Evidence of shortest-path policy — **PARTIAL** (code supports it, but repo lacks saved evidence artifacts)

### C) Task 2 — SARSA Level 1

- C1 Correct SARSA on-policy update (uses chosen next action) — **PASS**
- C2 Same exploration schedule as Q-learning — **PASS** (but shares same non-linear decay issue)
- C3 Evidence SARSA differs from Q-learning near hazards — **NOT DEMONSTRATED**

### D) Task 3 — Extend to Levels 2–3

- D1 Levels 2–3 include multiple apples, a key, and a chest — **PASS**
- D2 Both algorithms run correctly with correct termination/rewards — **LIKELY PASS**, but needs evidence plots/eval to be safe.

### E) Task 4 — Monster levels (4–5)

**Environment side:** PASS for basic mechanics (stochastic movement + death on contact).

**RL side:** tabular TD learning can handle stochastic transitions in principle.

**Evidence required:** training curves for levels 4 and 5 — **NOT PROVIDED**.

### F) Task 5 — Intrinsic reward (Level 6)

Spec requires:
- $r_i = \beta / \sqrt{n(s)+1}$
- env reward unchanged; intrinsic added for updates
- per-episode visit counter
- training curves with vs without intrinsic + short explanation

Current status:
- Visit counter per episode — **PASS**
- Formula — **FAIL**
- Env reward unchanged — **FAIL**
- With/without comparison curves — **NOT PROVIDED**

## 4) Rubric mapping to marks (from attached scheme)

### PART I-A
- A1: PASS
- A2: PASS with strict caveats

### PART I-B
- B1: PASS
- B2: PASS
- B3: FAIL
- B4: PASS
- B5: PARTIAL

### PART I-C
- C1: PASS
- C2: PASS (but impacted by B3 if “linear” is required here too)
- C3: FAIL / NOT DEMONSTRATED

### PART I-D
- D1: PASS
- D2: PARTIAL (needs evidence)

### PART I-F
- Intrinsic reward: FAIL (spec mismatch)
- Evidence: FAIL

## 5) Highest-impact improvement actions

1) **Make epsilon decay linear** and config-driven.
2) **Re-implement intrinsic reward exactly** to spec and keep env rewards unchanged.
3) **Add evidence generation**: plots and comparisons required by rubric.
4) Optionally clarify/adjust mechanics edge cases (only if allowed).

## 6) Suggested submission artifacts checklist

- Training curves:
  - Level 0 (Q-learning)
  - Level 1 (Q-learning vs SARSA comparison)
  - Levels 4 and 5 (learning curves)
  - Level 6 (with intrinsic vs without intrinsic)

- Policy evidence:
  - Rendered rollout video/gif/screenshots **or** evaluation table showing shortest path behavior.

- Short explanations:
  - SARSA vs Q-learning near hazards (on-policy vs off-policy).
  - Why intrinsic reward helps on Level 6 (exploration).
