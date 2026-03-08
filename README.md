# Mario Kart RL Project - Phase 2

## Overview
This project implements the Phase 2 skeleton for a reinforcement learning agent that will learn feature-based autonomous racing behavior in a custom Mario Kart environment.

## Baseline Algorithm
The baseline algorithm is Deep Q-Network (DQN).

## Project Structure
- `environment.py` - public environment entrypoint (auto-selects OS-specific backend)
- `environments/` - environment implementations
- `environments/windows.py` - Windows implementation (DME/GDB compatibility)
- `environments/mac.py` - macOS/Linux implementation (legacy GDB flow)
- `environments/legacy_gdb.py` - original legacy environment implementation
- `training_script.py` - startup verification and training loop entry point
- `dqn_agent.py` - DQN agent skeleton
- `q_network.py` - Q-network architecture
- `config.yaml` - hyperparameters and project settings
- `requirements.txt` - Python dependencies

## Setup
```bash
pip install -r requirements.txt
```

## Environment Verification 
Run a startup smoke test that initializes `MarioKartEnv`, calls `reset()`, runs a few `step()` calls, and writes captured output:

```bash
python training_script.py --iso "C:/path/to/Mario Kart.wbfs" --steps 5 --gfx Vulkan --output dqn_results/environment_verification.json
```

This creates a JSON report with reset/step outputs in `dqn_results/environment_verification.json`.
