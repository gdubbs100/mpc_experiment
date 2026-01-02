import pytest
import copy
from environment.dynamics_models import DynamicsModel
from environment.simulation_environment import OneDSimulationEnv
from environment.vehicle import Vehicle
from algo.mpc_agents import OracleMPCCEMAgent

from testing_utils import check_exact_equality

# Vehicle for test
vehicle = Vehicle(
    base_mass = 1.0,
    fuel_mass = 0.0,
    fuel_efficiency = 0.0,
    fuel_burn_rate=0.0,
    finite_fuel = False
)

# dynamics model for test
dynamics_model = DynamicsModel(
    resistance = 1.0,
    gravity = 1.0,
    landscape_func= lambda x: 1.0 * x, ## upwards slope
    time_increment = 1.0
)

# env for test
env = OneDSimulationEnv(
        target_location=2.0,
        dynamics_model = dynamics_model,
        vehicle=vehicle
    )

## agent for test
agent = OracleMPCCEMAgent(
    target_location = 2.0,
    dynamics_model=copy.deepcopy(dynamics_model),
    vehicle=copy.deepcopy(vehicle),
    num_lookahead_steps=5,
    num_rollouts=500,
    cem_iters = 10,
    cem_cutoff=0.5,
    initial_sampling_variance=100
)


