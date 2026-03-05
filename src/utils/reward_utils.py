import torch
import numpy as np

def calculate_distance_to_target(position: float, target: float) -> float:
    return abs(position - target)

def control_penalised_reward(absolute_distance_to_target: float, control_value: float) -> float:
    """
    distance to target is absolute value
    control value is between [-1, 1]
    """
    return absolute_distance_to_target**2 + control_value**2

def calculate_reward(position:float, target: float, control_value: float) -> float:
    return (position - target)**2 + control_value**2

def terminal_reward(position:float, velocity:float, target: float, control_value: float):
    """
    terminal reward repeats the final state reward + a penalty for velocity
    """
    return calculate_reward(position, target, control_value) + velocity**2

