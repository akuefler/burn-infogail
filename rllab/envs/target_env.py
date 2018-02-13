import numpy as np

import gym
from gym.spaces import Box
from gym.envs.classic_control.pendulum import PendulumEnv
from rllab.envs.gym_env import GymEnv

class NoisyPendulum(PendulumEnv):
    def __init__(self,k=3):
        super(NoisyPendulum, self).__init__()
        self._obs_space = self.observation_space
        low = np.concatenate([self._obs_space.low, -3.*np.ones(k)])
        high = np.concatenate([self._obs_space.high, 3.*np.ones(k)])
        self.observation_space = Box(low, high)
        self.k = k
        
    def reset(self):
        observation = super(NoisyPendulum,self).reset()
        return self.expand(observation,self.k)
        
    def step(self, action):
        observation, reward, done, info = super(NoisyPendulum,self).step(action)
        return self.expand(observation,self.k), reward, done, info
    
    @staticmethod
    def expand(observation, k):
        return np.concatenate([observation, np.random.normal(0,1,(k,))])
    
    @property
    def perfect_weights(self):
        return np.eye(self.k,3)

if __name__ == "__main__":
    gym.envs.register(
        id="NoisyPendulum-v0",
        entry_point='rllab.envs.target_env:NoisyPendulum',
        timestep_limit=999,
        reward_threshold=195.0,
    )
    env = GymEnv("NoisyPendulum-v0")
    
    env.reset()
