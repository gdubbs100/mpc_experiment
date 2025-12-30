import gymnasium as gym
import numpy as np

from environment.dynamics_models import DynamicsModel
from typing import Tuple, Optional

class OneDSimulationEnv(gym.Env):

    def __init__(
            self,
            target_location: float,
            dynamics_model: DynamicsModel,
            initial_position: float = 0.0,
            initial_velocity: float = 0.0,
            max_duration: int = 500,
            target_time_close_to_target: int = 5
        ):

        self.target_location = target_location
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.position = initial_position
        self.velocity = initial_velocity

        self.dynamics_model = dynamics_model
        dynamics_model.reset(
            initial_position=initial_position,
            initial_velocity=initial_velocity
        )
        self.max_duration = max_duration
        self.duration = 0
        self.time_close_to_target = 0
        self.target_time_close_to_target = target_time_close_to_target

        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape = (1,), dtype=float)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape = (2,), dtype=float)

    def calculate_distance_to_target(self, position: float) -> float:
        return np.abs(position - self.target_location)
    
    def calculate_reward(self):
        pass

    def get_observation(self) -> np.ndarray[float]:
        return np.array([self.position, self.velocity])

    def step(
            self, 
            action: float
        ) -> Tuple[float, float, bool, bool, dict]:
        self.position, self.velocity = self.dynamics_model.step(
            applied_force = action
        )
        observation = self.get_observation()

        ## get info
        info = dict()

        ## get reward
        distance_to_target = self.calculate_distance_to_target(position = self.position)
        reward = distance_to_target ## TODO: for now

        ## truncate after finite duration
        self.duration += 1
        if self.duration >= self.max_duration:
            truncated = True
        else:
            truncated = False

        ## termination - under what situation would we terminate?
        if distance_to_target < 1.0e-6:
            if self.time_close_to_target >= self.target_time_close_to_target:
                terminated = True
                info['success'] = 1
            else:
                self.time_close_to_target += 1
                info['success'] = 0
        else:
            self.time_close_to_target = 0
            terminated = False
            info['success'] = 0
        
        info['distance_to_target'] = distance_to_target
        info['time_close_to_target'] = self.time_close_to_target

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        self.position = self.initial_position
        self.velocity = self.initial_velocity
        observation = self.get_observation()

        self.time_close_to_target = 0
        info = dict()
        info['success'] = 0
        info['distance_to_target'] = self.calculate_distance_to_target(position = self.position)
        info['time_close_to_target'] = self.time_close_to_target
        
        return observation, info


