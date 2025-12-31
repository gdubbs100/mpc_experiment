import pytest
import numpy as np

from environment.dynamics_models import DynamicsModel
from environment.simulation_environment import OneDSimulationEnv
from environment.vehicle import Vehicle
from testing_utils import check_approximate_equality, check_exact_equality
# Vehicle for test
vehicle = Vehicle(
    base_mass = 1.0,
    fuel_mass = 0.0,
    fuel_efficiency = 0.0,
    finite_fuel = False
)
# dynamics model for test
dynamics_model = DynamicsModel(
    initial_position = 0.0,
    initial_velocity = 0.0,
    resistance = 1.0,
    gravity = 1.0,
    landscape_func= lambda x: 1.0 * x, ## upwards slope
    time_increment = 1.0
)

# env for test
env = OneDSimulationEnv(
        target_location=2.0,
        dynamics_model = dynamics_model,
        vehicle = vehicle
    )

def test_environment_step():
    acceleration = 1 - 1/np.sqrt(2)

    obs, reward, _, _, _ = env.step(action = 1.0)

    expected_position = env.initial_position + acceleration
    expected_velocity = env.initial_velocity + acceleration
    actual_position = obs[0]
    actual_velocity = obs[1]
    check_approximate_equality(expected_position, actual_position)
    check_approximate_equality(expected_velocity, actual_velocity)

    expected_reward = -np.abs(expected_position - env.target_location)
    actual_reward = reward
    check_approximate_equality(expected_reward, actual_reward)

def test_environment_reset():
    expected_position = env.initial_position
    expected_velocity = env.initial_velocity
    
    obs, _ = env.reset()

    actual_position = obs[0]
    actual_velocity = obs[1]
    check_approximate_equality(expected_position, actual_position)
    check_approximate_equality(expected_velocity, actual_velocity)

    actual_dynamics_model_position = env.dynamics_model.position
    actual_dynamics_model_velocity = env.dynamics_model.velocity

    ## also check dynamics model reset...
    check_exact_equality(expected_position, actual_dynamics_model_position)
    check_exact_equality(expected_velocity, actual_dynamics_model_velocity)