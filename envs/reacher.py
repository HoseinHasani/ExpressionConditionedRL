import numpy as np
import gym


class GoalReacher:
    def __init__(self, seed=0, max_pos=100, goal_rad=0.08, n_tasks=0,
                 step_length=8, max_step=500, start_goal_radius=50,
                 wind_power=0.3, neverending=False, disable_goal=True,
                 apply_noise=True, noise_power=0.005):
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        
        self.max_pos = max_pos
        self.step_length = step_length
        self.goal_rad = goal_rad * max_pos
        self.env_rad = max_pos
        self.wind_power = wind_power
        self.apply_noise = apply_noise
        self.noise_power = noise_power
        
        self.time = 0
        self.max_step = max_step
        self.episod_ind = 0
        self.neverending = neverending
        self.disable_goal = disable_goal
        
        self.task_id = None
        self.cur_pos = None
        self.start_pos = None
        self.goal_pos = None
        
        self.seed(seed)
        
        self.n_tasks = n_tasks
        self.task_vectors = self._generate_task_vectors() if self.n_tasks > 1 else [np.array([0.0, 0.0])]
        
        self.cur_state = self.reset()
        
        
    def seed(self, seed):
        np.random.seed(seed)
        
    def reset(self, seed=None, task_ind=None):
        if seed is not None:
            self.seed(seed)
        
        self.start_pos = self._random_position_within_radius(self.env_rad)
        if self.neverending:
            if self.cur_pos is None:
                self.cur_pos = self.start_pos
        else:
            self.cur_pos = self.start_pos

        self.goal_pos = self._random_position_within_radius(self.env_rad)
        
        if task_ind is None:
            self.task_id = np.random.randint(0, len(self.task_vectors))
        else:
            self.task_id = task_ind
            
            
        self.current_task_vector = self.task_vectors[self.task_id]
        self.cur_state = self._get_state(self.cur_pos, self.goal_pos)
        self.time = 0

        return self.cur_state, None
    
    def _random_position_within_radius(self, radius):
        angle = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, radius)
        x = r * np.cos(angle)
        y = r * np.sin(angle)
        return np.array([x, y])
        
            
    def _generate_task_vectors(self):
        task_vectors = []
        angle_increment = 2 * np.pi / self.n_tasks
        for i in range(self.n_tasks):
            angle = i * angle_increment
            vector = np.array([np.cos(angle), np.sin(angle)])
            task_vectors.append(vector)
        return task_vectors
    
    def step(self, act):
        dx = self.step_length * act[0]
        dy = self.step_length * act[1]
        base_step = np.array([dx, dy])
        
        noise_step = np.zeros_like(base_step)
        
        if self.apply_noise:
            noise_step = self.step_length * np.random.normal(0, self.noise_power, 2)
        
        task_step = self.wind_power * self.step_length * self.current_task_vector
        
        new_pos = self.cur_pos + base_step + task_step + noise_step
        
        if np.linalg.norm(new_pos) < 10.2 * self.max_pos:
            self.cur_pos = new_pos
            
        self.cur_state = self._get_state(self.cur_pos, self.goal_pos)
        
        reward, done = self.get_reward(self.cur_state, act)
        self.time += 1
        
        if self.time > self.max_step:
            done = True
        
        reward_dict = {'r_total': reward}
        env_info = {'ob': self.cur_state,
                    'rewards': reward_dict,
                    'score': reward}
                
        return self.cur_state.copy(), reward, done, done, env_info
    
    def get_reward(self, state, action):
        pos = state[:2] * self.max_pos
        goal = state[2:] * self.max_pos
        
        distance_to_goal = np.linalg.norm(pos - goal)
        reward = - distance_to_goal / self.max_pos
        done = distance_to_goal < self.goal_rad 

        if self.disable_goal:
            done = False
            
        
        reward -= 0.1 * 1 #self.time
        
        if done:
            reward += 1.0  
            
        return reward, done
            
        
    def _get_state(self, pos, goal):
        return np.concatenate([pos / self.max_pos, goal / self.max_pos])
    
    def close(self):
        pass