import gymnasium as gym
import numpy as np

class ConditionalStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, context_size=7, trajectory_limit=100):
        super().__init__(env)

        self.context_size = context_size
        self.context = np.zeros(context_size, dtype=np.float32)
        
        self.trajectory_limit = trajectory_limit
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
        next_obs, reward, done, info = self.env.step(action)

        state = self.env.state
        self._add_to_buffer((state, action, reward))

        return self.observation(next_obs), reward, done, info

    def reset(self, **kwargs):
        self.trajectory_buffer = []
        initial_obs = self.env.reset(**kwargs)
        return self.observation(initial_obs)

    def _add_to_buffer(self, entry):
        self.trajectory_buffer.append(entry)
        if len(self.trajectory_buffer) > self.trajectory_limit:
            self.trajectory_buffer.pop(0)  
            