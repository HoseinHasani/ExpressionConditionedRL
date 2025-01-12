import gymnasium as gym
import numpy as np

class ConditionalStateWrapper(gym.ObservationWrapper):
    def __init__(self, env, context_size=7):
        super().__init__(env)
        
        self.context_size = context_size
        self.context = np.zeros(context_size, dtype=np.float32)
        
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
