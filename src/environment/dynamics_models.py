import numpy as np
from typing import Callable, Tuple
from .environment_utils import numerical_grad

class DynamicsModel:

    def __init__(
            self, 
            initial_position: float,
            initial_velocity: float,
            resistance: float, 
            mass: float, 
            gravity: float,
            landscape_func: Callable[[float], float],
            time_increment = float
        ):
        self.position = initial_position
        self.velocity = initial_velocity
        self.resistance = resistance
        self.mass = mass
        self.gravity = gravity
        self.landscape_func = landscape_func
        self.time_increment = time_increment

    def reset(self, initial_position: float, initial_velocity: float) -> None:
        self.position = initial_position
        self.velocity = initial_velocity

    def update_position(self, applied_force: float) -> Tuple[float, float]:
        acceleration = self.calculate_acceleration(
            applied_force = applied_force,
            current_velocity = self.velocity,
            current_position = self.position
        )
        next_velocity = self.velocity + acceleration * self.time_increment
        next_position = self.position + next_velocity * self.time_increment
        return next_position, next_velocity
    
    def calculate_acceleration(
            self, 
            applied_force: float, 
            current_velocity: float, 
            current_position: float
        ) -> float:
        friction = self.resistance * current_velocity
        force_from_gravity = self.calculate_force_from_gravity(
            current_position = current_position
        )
        return (force_from_gravity + applied_force - friction) / self.mass
    
    def calculate_force_from_gravity(self, current_position: float) -> float:
        s = numerical_grad(
            landscape_func=self.landscape_func,
            x = current_position
        )
        return -self.mass * self.gravity * (s / np.sqrt(1 + s**2))
    
    def step(self, applied_force: float) -> Tuple[float, float]:
        self.position, self.velocity = self.update_position(
            applied_force = applied_force
        )
        return self.position, self.velocity



