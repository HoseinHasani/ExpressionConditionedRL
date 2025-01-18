# Expression Conditioned Reinforcement Learning

This project implements a framework for **Expression Conditioned Reinforcement Learning** (ECRL), designed to adapt and perform in non-stationary environments using task inference mechanisms. The project utilizes environments such as `HalfCheetah-v4`, `Pendulum-v1`, `Swimmer-v4`, `Reacher-v4`, and `CartPole`, enabling the creation of dynamic, conditioned scenarios. The framework is built on top of **Stable-Baselines3**, leveraging its algorithms for training and evaluation.

## Features
- **Non-Stationary Environment Wrapper**: Dynamically modifies environmental parameters such as gravity, mass, friction, or viscosity.
- **Task Inference Modules**: Supports simple inference (`SimpleTaskInference`) and symbolic regression-based inference (`SymbolicRegressionInference`).
- **Visualization Utilities**: Generates performance plots with moving average rewards for evaluation.
- **Customizable via CLI**: Easily switch between environments and configurations using command-line arguments.

## Requirements

To get started, ensure the following dependencies are installed:

- Stable-Baselines3
- Gymnasium
- Matplotlib
- Pandas

Install required packages via:

```bash
pip install -r requirements.txt
```

## Project Structure

```plaintext
.
├── main.py                   # Main script to train and evaluate models
├── env_utils                 # Contains environment wrappers and utilities
│   ├── conditioned_envs.py   # Wrapper for conditional states
│   └── non_stationary_env.py # Non-stationary environment implementation
├── task_inference_utils      # Modules for task inference
│   ├── simple_inference.py   # Simple task inference implementation
│   ├── base_inference.py     # Simple task inference implementation
│   └── sr_inference.py       # Symbolic regression inference implementation
├── visualization_utils.py    # Utilities for loading data and visualization
├── requirements.txt          # Dependency file
└── README.md                 # Project documentation
```

## Usage

### Training a Model
Run the `main.py` script to train a reinforcement learning model. Use the `--env` argument to select the environment:

```bash
python main.py --env HalfCheetah-v4
```

Other available environments:
- `Pendulum-v1`
- `Swimmer-v4`
- `Reacher-v4`
- `CartPole`

### Visualizing Results
The script automatically saves performance logs to the `./sac_logs/` directory. Use the built-in visualization tools to analyze results:

```python
from visualization_utils import load_monitor_data, plot_moving_average_reward

sac_df = load_monitor_data('./sac_logs/')
plot_moving_average_reward(sac_df, title='Performance on HalfCheetah', label='SAC', color='blue')
```

## Examples

### Training with HalfCheetah-v4
```bash
python main.py --env HalfCheetah-v4
```

### Training with Pendulum-v1
```bash
python main.py --env Pendulum-v1
```

## Customization

### Adding a New Environment
To add a custom environment:
1. Define the environment using OpenAI Gym API.
2. Update the `NonStationaryEnv` wrapper to handle specific dynamic changes for the new environment.
3. Register the environment in `main.py`.

### Modifying Visualization
Enhance or customize visualizations by editing `visualization_utils.py`.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


