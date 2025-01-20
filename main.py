import os
import gymnasium
import pandas as pd
from stable_baselines3 import SAC, DDPG, DQN, PPO, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env_utils.conditioned_envs import ConditionalStateWrapper
from env_utils.non_stationary_wrapper import NonStationaryEnv
from envs.reacher import GoalReacherEnv
from task_inference_utils.simple_inference import SimpleTaskInference
from task_inference_utils.sr_inference import SymbolicRegressionInference
from task_inference_utils.vae_inference import VAEInference
from visualization_utils import load_monitor_data, plot_moving_average_reward
import argparse

# Hyperparameters
max_ep_len = 220
n_tasks = 3
task_name = None

total_timesteps = 7000
learning_rate = 0.001


# Argument parser
parser = argparse.ArgumentParser(description="Expression Conditioned Reinforcement Learning")
parser.add_argument('--env', type=str, default='GoalReacher',
                    choices=['HalfCheetah-v4', 'Pendulum-v1', 'Swimmer-v4', 'Reacher-v4', 'CartPole-v1', 'GoalReacher'],
                    help='Environment to use for training and evaluation.')
parser.add_argument('--algo', type=str, default='SAC',
                    choices=['SAC', 'DDPG', 'DQN', 'PPO', 'TD3'],
                    help='Reinforcement learning algorithm to use.')
parser.add_argument('--inference', type=str, default='sr',
                    choices=['simple', 'vae', 'sr'],
                    help='Task inference method to use.')
args = parser.parse_args()


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

env = GoalReacherEnv()  if args.env == 'GoalReacher' else gymnasium.make(args.env)
env = NonStationaryEnv(env, max_ep_len, n_tasks, task_name, args.env)
env = ConditionalStateWrapper(env, task_inference=task_inference)
env = Monitor(env, log_dir)

# Select RL algorithm
if args.algo == 'SAC':
    model = SAC('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'DDPG':
    model = DDPG('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'DQN':
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'PPO':
    model = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate)
elif args.algo == 'TD3':
    model = TD3('MlpPolicy', env, verbose=1, learning_rate=learning_rate)

# Evaluation environment
eval_env = GoalReacherEnv()  if args.env == 'GoalReacher' else gymnasium.make(args.env)  
eval_env = NonStationaryEnv(eval_env, max_ep_len, n_tasks, task_name, args.env)
eval_env = ConditionalStateWrapper(eval_env, task_inference=task_inference)
eval_env = Monitor(eval_env, log_dir)

# Callback for evaluation
eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                             log_path=log_dir, eval_freq=50,
                             deterministic=True, render=False)

# Train the model
model.learn(total_timesteps=total_timesteps, callback=eval_callback)

# Load and visualize results
results_df = load_monitor_data(log_dir)
plot_moving_average_reward(results_df, title=title, label=args.algo, color='blue')
