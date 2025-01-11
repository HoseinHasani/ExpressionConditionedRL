import numpy as np
import pandas as pd
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




sac_df = pd.read_csv(os.path.join(sac_log_dir, 'monitor.csv'), skiprows=1)
print("SAC Results:")
print(sac_df.head())


sac_df['r'] = pd.to_numeric(sac_df['r'], errors='coerce')
sac_df['l'] = pd.to_numeric(sac_df['l'], errors='coerce')
sac_df['t'] = pd.to_numeric(sac_df['t'], errors='coerce')
sac_df = sac_df.dropna()


sac_df['moving_avg'] = sac_df['r'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))

plt.plot(sac_df.index, sac_df['moving_avg'], label='SAC', color='blue')

plt.title('SAC Performance on HalfCheetah')
plt.xlabel('Episode Index')
plt.ylabel('Moving Average Reward')
plt.legend()
plt.grid()
plt.show()



