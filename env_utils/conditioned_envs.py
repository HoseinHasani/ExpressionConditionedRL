import gymnasium as gym
import numpy as np

class ConditionalStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, context_size=7, trajectory_limit=100):
        super().__init__(env)

        self.context_size = context_size
        self.context = np.zeros(context_size, dtype=np.float32)
        
        self.buffer_size = trajectory_limit
        self.trajectory_buffer = []  

        obs_space = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([obs_space.low, -np.inf * np.ones(context_size)]),
            high=np.concatenate([obs_space.high, np.inf * np.ones(context_size)]),
            dtype=np.float32
        )

    def observation(self, obs):
        return np.concatenate([obs, self.context])

    def set_context(self, context):
        assert len(context) == self.context_size, \
            f"Context must have size {self.context_size}, but got size {len(context)}."
        self.context = np.array(context, dtype=np.float32)

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
    
        self.trajectory_buffer.append((self.current_obs, action, reward))
        if len(self.trajectory_buffer) > self.buffer_size:
            self.trajectory_buffer.pop(0)
    
        self.current_obs = next_obs  
        return self.observation(next_obs), reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.trajectory_buffer = []
        initial_obs, _ = self.env.reset(**kwargs)
        self.current_obs = initial_obs
        return self.observation(initial_obs), {}

    def _add_to_buffer(self, entry):
        self.trajectory_buffer.append(entry)
        if len(self.trajectory_buffer) > self.trajectory_limit:
            self.trajectory_buffer.pop(0)  
            