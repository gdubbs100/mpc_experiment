import pytest
import copy
import numpy as np

from environment.dynamics_models import DynamicsModel
from environment.simulation_environment import OneDSimulationEnv
from environment.vehicle import Vehicle
from algo.mpc_agents import (
    OracleCEMAgent, 
    OracleRandomShootingAgent, 
    OracleMPPIAgent,
    OracleMPCAgent
)

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


def test_mpc_iter():
    agent = OracleMPCAgent(
            target_location =2.0, 
            dynamics_model = copy.deepcopy(dynamics_model),
            vehicle = copy.deepcopy(vehicle),
            num_lookahead_steps = 5, 
            num_rollouts = 10, 
    )

    state = np.array([1.0, 1.0, 1.0])
    actions = np.ones((10, 5))

    rewards = agent.run_mpc_iter(initial_state=state, actions = actions)
    assert rewards.shape == (10,)

def test_rollout_reward():
    lookahead_steps = 5
    agent = OracleMPCAgent(
        target_location =2.0, 
        dynamics_model = copy.deepcopy(dynamics_model),
        vehicle = copy.deepcopy(vehicle),
        num_lookahead_steps = lookahead_steps, 
        num_rollouts = 10, 
    )

    rollout_reward = agent.rollout(
        initial_position=0.0,
        initial_velocity=0.0,
        remaining_fuel=0.0,
        actions = np.ones((lookahead_steps))
    )
    assert isinstance(rollout_reward, float), f"Rollout reward should be scalar float, found: {type(rollout_reward)}"

def test_random_shooting_agent():
    agent = OracleRandomShootingAgent(
        target_location=1.0,
        dynamics_model=copy.deepcopy(dynamics_model),
        vehicle=copy.deepcopy(vehicle),
        num_lookahead_steps=10,
        num_rollouts = 10
    )

    state = np.array([1.0,1.0, 1.0])
    u = agent.act(state = state)
    assert abs(u) < 1, f"value of u not within +/-1. Actual value: {u}"
    assert isinstance(u, float), f"u should be float. Actual type: {type(u)}"

def test_mppi_agent_weight_dims():
    agent = OracleMPPIAgent(
            target_location =2.0, 
            dynamics_model = copy.deepcopy(dynamics_model),
            vehicle = copy.deepcopy(vehicle),
            num_lookahead_steps = 5, 
            num_rollouts = 10, 
            learning_iters=2,
            temperature = 1.0
    )

    dummy_rewards = np.ones((10, ))
    weights = agent.compute_mppi_weights(rewards = dummy_rewards)
    assert weights.shape == (10, )

def test_mppi_actions():
    agent = OracleMPPIAgent(
            target_location =2.0, 
            dynamics_model = copy.deepcopy(dynamics_model),
            vehicle = copy.deepcopy(vehicle),
            num_lookahead_steps = 5, 
            num_rollouts = 10, 
            learning_iters=2,
            temperature = 1.0
    )

    state = np.array([1.0, 1.0, 1.0])

    u = agent.act(state = state)
    
    assert abs(u) < 1, f"value of u not within +/-1. Actual value: {u}"
    assert isinstance(u, float), f"u should be float. Actual type: {type(u)}"

def test_cem_actions():
    agent = OracleCEMAgent(
            target_location =2.0, 
            dynamics_model = copy.deepcopy(dynamics_model),
            vehicle = copy.deepcopy(vehicle),
            num_lookahead_steps = 5, 
            num_rollouts = 10, 
            learning_iters=2,
            cem_cutoff=0.95,
            initial_sampling_variance=1.0
    )

    state = np.array([1.0, 1.0, 1.0])

    u = agent.act(state = state)
    
    assert abs(u) < 1, f"value of u not within +/-1. Actual value: {u}"
    assert isinstance(u, float), f"u should be float. Actual type: {type(u)}"