import os
import gymnasium
import argparse
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from env_utils.conditioned_envs import ConditionalStateWrapper
from task_inference_utils.simple_inference import SimpleTaskInference
from task_inference_utils.sr_inference import SymbolicRegressionInference
from visualization_utils import load_monitor_data, plot_moving_average_reward

def main():
    parser = argparse.ArgumentParser(description="Run RL experiment on different environments.")
    parser.add_argument("--env", type=str, choices=["HalfCheetah-v4", "Pendulum-v1", "Swimmer-v4", "Reacher-v4", "cartpole"], 
                        required=True, help="Choose the environment.")
    args = parser.parse_args()

    sac_log_dir = './sac_logs/'
    os.makedirs(sac_log_dir, exist_ok=True)

    task_inference = SimpleTaskInference(14) 
    # task_inference = SymbolicRegressionInference(context_size=14) 

    # Create environment
    env = gymnasium.make(args.env)
    env = ConditionalStateWrapper(env, task_inference=task_inference)

    sac_model = SAC('MlpPolicy', env, verbose=1, learning_rate=0.001)

    # Evaluation environment
    eval_env = gymnasium.make(args.env)
    eval_env = ConditionalStateWrapper(eval_env, task_inference=task_inference)
    eval_env = Monitor(eval_env, sac_log_dir)

    eval_callback_sac = EvalCallback(eval_env, best_model_save_path=sac_log_dir,
                                     log_path=sac_log_dir, eval_freq=50,
                                     deterministic=True, render=False)

    # Train the model
    sac_model.learn(total_timesteps=7000, callback=eval_callback_sac)

    # Load and visualize results
    sac_df = load_monitor_data(sac_log_dir)
    plot_moving_average_reward(sac_df, title=f"SAC Performance on {args.env}", label='SAC', color='blue')

if __name__ == "__main__":
    main()
