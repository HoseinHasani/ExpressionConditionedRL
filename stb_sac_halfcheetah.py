import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium

sac_log_dir = './sac_logs/'
os.makedirs(sac_log_dir, exist_ok=True)


env = make_vec_env('HalfCheetah-v4', n_envs=4, wrapper_class=lambda e: Monitor(e, sac_log_dir))

sac_model = SAC('MlpPolicy', env, verbose=1, learning_rate=0.001)
eval_env = gymnasium.make("HalfCheetah-v4")

eval_callback_sac = EvalCallback(eval_env, best_model_save_path='./sac_logs/',
                             log_path='./sac_logs/', eval_freq=100,
                             deterministic=True, render=False)
sac_model.learn(total_timesteps=10000, callback=eval_callback_sac)

