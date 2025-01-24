import os
import sys
import json
import gymnasium
import numpy as np
from datetime import datetime

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from env_utils.conditioned_envs import ConditionalStateWrapper
from env_utils.non_stationary_wrapper import NonStationaryEnv
from envs.goal_reacher import GoalReacherEnv
from task_inference_utils.simple_inference import SimpleTaskInference
from task_inference_utils.sr_inference import SymbolicRegressionInference
from task_inference_utils.vae_inference import VAEInference
from general_utils import fix_seed
import argparse

fix_seed(seed=0)

parser = argparse.ArgumentParser(description="Gather data for task inference evaluation")
parser.add_argument('--env', type=str, default='GoalReacher',
                    choices=['HalfCheetah-v4', 'Pendulum-v1', 'Swimmer-v4', 'Reacher-v4', 'CartPole-v1', 'GoalReacher'],
                    help='Environment to use for data gathering.')
parser.add_argument('--inference', type=str, default='sr',
                    choices=['simple', 'vae', 'sr'],
                    help='Task inference method to use.')
parser.add_argument('--output', type=str, default='data/task_inference_data.json',
                    help='Path to save the gathered data.')
parser.add_argument('--episodes', type=int, default=10, help='Number of episodes for data gathering.')
args = parser.parse_args()

config_path = os.path.join('../configs', f"{args.env}.json")
with open(config_path, 'r') as f:
    config = json.load(f)

max_ep_len = config['max_ep_len']
n_tasks = config['n_tasks']
task_name = config['task_name']

os.makedirs(os.path.dirname(args.output), exist_ok=True)

if args.inference == 'simple':
    task_inference = SimpleTaskInference(14)
elif args.inference == 'vae':
    task_inference = VAEInference()
elif args.inference == 'sr':
    task_inference = SymbolicRegressionInference(context_size=14)

env = GoalReacherEnv() if args.env == 'GoalReacher' else gymnasium.make(args.env)
env = NonStationaryEnv(env, max_ep_len, n_tasks, task_name, args.env)
env = ConditionalStateWrapper(env, task_inference=task_inference)

data = []
for episode_id in range(args.episodes):
    print(f"{datetime.now()} - Gathering data for episode {episode_id}")
    obs, _ = env.reset()
    terminated = False
    truncated = False
    episode_conditions = []

    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        condition = info.get('context', None)  
        if condition is not None:
            episode_conditions.append(condition)

    data.append({
        'episode_id': episode_id,
        'conditions': episode_conditions
    })

metadata = {
    'env': args.env,
    'inference': args.inference,
    'num_episodes': args.episodes
}
with open(args.output, 'w') as f:
    json.dump({'metadata': metadata, 'data': data}, f, indent=4)

print(f"Data gathering complete. Saved to {args.output}")
