import os
import argparse
import gymnasium
import pandas as pd
from stable_baselines3 import SAC, DDPG, DQN, TD3, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env_utils.conditioned_envs import ConditionalStateWrapper
from task_inference_utils.simple_inference import SimpleTaskInference
from visualization_utils import load_monitor_data, plot_moving_average_reward

# Parse arguments
parser = argparse.ArgumentParser(description="Run a reinforcement learning experiment with various algorithms and environments.")
parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                    choices=['HalfCheetah-v4', 'Pendulum-v1', 'Swimmer-v4', 'Reacher-v4', 'CartPole-v1'],
                    help='Environment to use (default: HalfCheetah-v4).')
parser.add_argument('--algo', type=str, default='SAC',
                    choices=['SAC', 'DDPG', 'DQN', 'PPO', 'TD3'],
                    help='Algorithm to use (default: SAC).')
args = parser.parse_args()

# Set up environment and algorithm based on arguments
env_name = args.env
algo_name = args.algo
log_dir = f'./logs/{algo_name}_{env_name}/'
os.makedirs(log_dir, exist_ok=True)

# Task inference setup
task_inference = SimpleTaskInference(14)

# Environment setup
env = gymnasium.make(env_name)
env = ConditionalStateWrapper(env, task_inference=task_inference)

# Select algorithm
if algo_name == 'SAC':
    model = SAC('MlpPolicy', env, verbose=1, learning_rate=0.001)
elif algo_name == 'DDPG':
    model = DDPG('MlpPolicy', env, verbose=1, learning_rate=0.001)
elif algo_name == 'DQN':
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.001)
elif algo_name == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.001)
elif algo_name == 'TD3':
    model = TD3('MlpPolicy', env, verbose=1, learning_rate=0.001)
else:
    raise ValueError(f"Unsupported algorithm: {algo_name}")

# Evaluation environment setup
eval_env = gymnasium.make(env_name)
eval_env = ConditionalStateWrapper(eval_env, task_inference=task_inference)
eval_env = Monitor(eval_env, log_dir)

# Evaluation callback setup
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                              log_path=log_dir, eval_freq=50,
                              deterministic=True, render=False)

# Train the model
model.learn(total_timesteps=7000, callback=eval_callback)

# Load and visualize results
results_df = load_monitor_data(log_dir)
plot_title = f'{algo_name} Performance on {env_name}'
plot_moving_average_reward(results_df, title=plot_title, label=algo_name, color='blue')

