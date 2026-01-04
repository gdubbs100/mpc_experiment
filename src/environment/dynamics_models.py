import numpy as np
from typing import Callable, Tuple
from utils.environment_utils import numerical_grad

class DynamicsModel:

    def __init__(
            self, 
            resistance: float, 
            gravity: float,
            landscape_func: Callable[[float], float],
            time_increment: float
        ):
        self.resistance = resistance
        self.gravity = gravity
        self.landscape_func = landscape_func
        self.time_increment = time_increment


    def update_position(
            self, 
            applied_force: float, 
            mass: float, 
            current_position: float,
            current_velocity: float,
        ) -> Tuple[float, float]:
        acceleration = self.calculate_acceleration(
            applied_force = applied_force,
            current_velocity = current_velocity,
            current_position = current_position,
            mass = mass
        )
        next_velocity = current_velocity + acceleration * self.time_increment
        next_position = current_position + next_velocity * self.time_increment
        return next_position, next_velocity
    
    def calculate_acceleration(
            self, 
            applied_force: float, 
            current_position: float,
            current_velocity: float, 
            mass: float
        ) -> float:
        friction = self.resistance * current_velocity
        force_from_gravity = self.calculate_force_from_gravity(
            current_position = current_position,
            mass = mass
        )
        return (force_from_gravity + applied_force - friction) / mass
    
    def calculate_force_from_gravity(self, current_position: float, mass: float) -> float:
        s = numerical_grad(
            landscape_func=self.landscape_func,
            x = current_position
        )
        return -mass * self.gravity * (s / np.sqrt(1 + s**2))

    def step(
            self, 
            applied_force: float, 
            mass: float,
            current_position: float,
            current_velocity: float
        ) -> Tuple[float, float]:
        position, velocity = self.update_position(
            applied_force = applied_force,
            mass = mass,
            current_position=current_position,
            current_velocity=current_velocity
        )
        return position, velocity



