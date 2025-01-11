import gym
from stable_baselines3 import SAC

env = gym.make('HalfCheetah-v3')

model = SAC('MlpPolicy', env, verbose=1)

model.learn(total_timesteps=10000)

model.save("sac_half_cheetah")

obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
