import os
import sys
import gymnasium
import numpy as np
import json
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
subtract_baseline = False

parser = argparse.ArgumentParser(description="Gather data for task inference evaluation")
parser.add_argument('--env', type=str, default='Swimmer-v4',
                    choices=['HalfCheetah-v4', 'Pendulum-v1', 'Swimmer-v4', 'Reacher-v4', 'CartPole-v1', 'GoalReacher'],
                    help='Environment to use for data gathering.')
parser.add_argument('--inference', type=str, default='sr',
                    choices=['simple', 'vae', 'sr'],
                    help='Task inference method to use.')
parser.add_argument('--output_dir', type=str, default='data',
                    help='Directory to save the gathered data.')
# parser.add_argument('--episodes', type=int, default=32*8, help='Number of episodes for data gathering.')

args = parser.parse_args()

config_path = os.path.join('../configs', f"{args.env}.json")
with open(config_path, 'r') as f:
    config = json.load(f)

max_ep_len = config['max_ep_len']
n_tasks = config['n_tasks']
task_name = config['task_name']
context_size = config['context_size']

output_path = os.path.join(args.output_dir, f"{args.env}_{args.inference}")
os.makedirs(output_path, exist_ok=True)


if args.env == 'Reacher-v4':
    n_episodes = 132
if args.env == 'HalfCheetah-v4':
    n_episodes = 10
if args.env == 'Swimmer-v4':
    n_episodes = 9
if args.env == 'Pendulum-v1':
    n_episodes = 25
if args.env == 'CartPole-v1':
    n_episodes = 200

if args.inference == 'simple':
    task_inference = SimpleTaskInference(context_size)
elif args.inference == 'vae':
    task_inference = VAEInference()
elif args.inference == 'sr':
    task_inference = SymbolicRegressionInference(context_size=context_size)

if subtract_baseline:
    baseline_env = GoalReacherEnv() if args.env == 'GoalReacher' else gymnasium.make(args.env)
    baseline_env = ConditionalStateWrapper(baseline_env, task_inference=task_inference)

    baseline_conditions = []
    for _ in range(n_tasks):
        obs, _ = baseline_env.reset()
        terminated, truncated = False, False

        while not terminated and not truncated:
            action = baseline_env.action_space.sample()
            obs, reward, terminated, truncated, info = baseline_env.step(action)
            condition = info.get('context', None)
            if condition is not None:
                baseline_conditions.append(condition)

    baseline_conditions = np.array(baseline_conditions)
    baseline_vector = np.mean(baseline_conditions, axis=0)
else:
    baseline_vector = np.zeros(context_size, dtype=np.float32)
    

env = GoalReacherEnv() if args.env == 'GoalReacher' else gymnasium.make(args.env)
env = NonStationaryEnv(env, max_ep_len, n_tasks, task_name, args.env)
env = ConditionalStateWrapper(env, task_inference=task_inference)

conditions = []
labels = []
counter = 0
prev_task = 0

for episode_id in range(n_episodes):
    print(f"{datetime.now()} - Gathering data for episode {episode_id}")
    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        condition = info.get('context', None)
        if condition is not None:
            condition = condition - baseline_vector
            cur_task = env.current_task
            if cur_task == prev_task:
                conditions.append(condition)
                labels.append(cur_task)
                prev_task = cur_task
                counter = 0
            else:
                counter += 1
                if counter > 25 or args.env == 'CartPole-v1':
                    prev_task = cur_task
                    
        if steps > 600:
            break
                    
    
conditions = np.array(conditions)
labels = np.array(labels)

np.save(os.path.join(output_path, 'conditions.npy'), conditions)
np.save(os.path.join(output_path, 'labels.npy'), labels)

print(f"Data gathering complete. Saved conditions and labels in {output_path}")
