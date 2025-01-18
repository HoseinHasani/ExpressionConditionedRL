import os
import gymnasium
import pandas as pd
from stable_baselines3 import SAC, DDPG, DQN, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env_utils.conditioned_envs import ConditionalStateWrapper
from task_inference_utils.simple_inference import SimpleTaskInference
from task_inference_utils.sr_inference import SymbolicRegressionInference
from task_inference_utils.vae_inference import VAEInference
from visualization_utils import load_monitor_data, plot_moving_average_reward
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Expression Conditioned Reinforcement Learning")
parser.add_argument('--env', type=str, default='HalfCheetah-v4',
                    choices=['HalfCheetah-v4', 'Pendulum-v1', 'Swimmer-v4', 'Reacher-v4', 'CartPole-v1'],
                    help='Environment to use for training and evaluation.')
parser.add_argument('--algo', type=str, default='SAC',
                    choices=['SAC', 'DDPG', 'DQN', 'PPO', 'TD3'],
                    help='Reinforcement learning algorithm to use.')
parser.add_argument('--inference', type=str, default='simple',
                    choices=['simple', 'vae', 'sr'],
                    help='Task inference method to use.')
args = parser.parse_args()

# Define log directory and titles based on algorithm and environment
log_dir = f'./logs/{args.algo.lower()}_{args.env.lower()}/'
os.makedirs(log_dir, exist_ok=True)

title = f'{args.algo} Performance on {args.env}'

# Select task inference method
if args.inference == 'simple':
    task_inference = SimpleTaskInference(14)
elif args.inference == 'vae':
    task_inference = VAEInference()
elif args.inference == 'sr':
    task_inference = SymbolicRegressionInference(context_size=14)

# Create environment
env = gymnasium.make(args.env)
env = ConditionalStateWrapper(env, task_inference=task_inference)
env = Monitor(env, log_dir)

# Select RL algorithm
if args.algo == 'SAC':
    model = SAC('MlpPolicy', env, verbose=1, learning_rate=0.001)
elif args.algo == 'DDPG':
    model = DDPG('MlpPolicy', env, verbose=1, learning_rate=0.001)
elif args.algo == 'DQN':
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.001)
elif args.algo == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.001)
elif args.algo == 'TD3':
    model = TD3('MlpPolicy', env, verbose=1, learning_rate=0.001)

# Evaluation environment
eval_env = gymnasium.make(args.env)
eval_env = ConditionalStateWrapper(eval_env, task_inference=task_inference)
eval_env = Monitor(eval_env, log_dir)

# Callback for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=50,
                             deterministic=True, render=False)

# Train the model
model.learn(total_timesteps=7000, callback=eval_callback)

# Load and visualize results
results_df = load_monitor_data(log_dir)
plot_moving_average_reward(results_df, title=title, label=args.algo, color='blue')
