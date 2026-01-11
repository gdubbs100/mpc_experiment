import gymnasium as gym
import numpy as np

from environment.dynamics_models import DynamicsModel
from environment.vehicle import Vehicle
from typing import Tuple, Optional
from utils.reward_utils import calculate_reward

class OneDSimulationEnv(gym.Env):

    def __init__(
            self,
            target_location: float,
            dynamics_model: DynamicsModel,
            vehicle: Vehicle,
            initial_position: float = 0.0,
            initial_velocity: float = 0.0,
            max_duration: int = 500,
            target_time_close_to_target: int = 10,
            distance_to_target_threshold: float = 0.01,
            velocity_threshold: float = 0.01
        ):

        self.target_location = target_location
        self.initial_position = initial_position
        self.initial_velocity = initial_velocity
        self.position = initial_position
        self.velocity = initial_velocity

        self.vehicle = vehicle
        self.initial_fuel_mass = self.vehicle.fuel_mass

        self.dynamics_model = dynamics_model
        self.max_duration = max_duration
        self.duration = 0
        self.time_close_to_target = 0
        self.target_time_close_to_target = target_time_close_to_target
        self.distance_to_target_threshold = distance_to_target_threshold
        self.velocity_threshold = velocity_threshold

        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape = (1,), dtype=float)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape = (2,), dtype=float)

    def calculate_distance_to_target(self, position: float) -> float:
        return np.abs(position - self.target_location)
    
    def calculate_reward(self):
        pass

    def get_observation(self) -> np.ndarray[float]:
        return np.array([self.position, self.velocity, self.vehicle.fuel_mass])

    def step(
            self, 
            action: float
        ) -> Tuple[float, float, bool, bool, dict]:

        applied_force = self.vehicle.generate_force(
            u = action, 
            time_increment=self.dynamics_model.time_increment
        )
        self.position, self.velocity = self.dynamics_model.step(
            applied_force = applied_force,
            mass = self.vehicle.mass,
            current_position = self.position,
            current_velocity = self.velocity
        )
        observation = self.get_observation()

        ## get info
        info = dict()

        ## get reward
        distance_to_target = self.calculate_distance_to_target(position = self.position)
        reward = -calculate_reward(
            position = self.position,
            target = self.target_location,
            control_value = action
        )
        ## truncate after finite duration
        truncated = False
        self.duration += 1
        if self.duration >= self.max_duration:
            truncated = True

        terminated = False
        info['success'] = 0

        if (distance_to_target < self.distance_to_target_threshold) and (abs(self.velocity) < self.velocity_threshold):
            if self.time_close_to_target >= self.target_time_close_to_target:
                terminated = True
                info['success'] = 1
            else:
                self.time_close_to_target += 1
        else:
            self.time_close_to_target = 0
            
        
        info['distance_to_target'] = distance_to_target
        info['time_close_to_target'] = self.time_close_to_target

        return observation, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)
        self.duration = 0
        self.position = self.initial_position
        self.velocity = self.initial_velocity
        self.vehicle.reset(fuel_mass = self.initial_fuel_mass)
        observation = self.get_observation()

        self.time_close_to_target = 0
        info = dict()
        info['success'] = 0
        info['distance_to_target'] = self.calculate_distance_to_target(position = self.position)
        info['time_close_to_target'] = self.time_close_to_target
        
        return observation, info


