import pytest
import copy
from environment.dynamics_models import DynamicsModel
from environment.simulation_environment import OneDSimulationEnv
from algo.mpc_agents import OracleMPCCEMAgent

from testing_utils import check_exact_equality

# dynamics model for test
dynamics_model = DynamicsModel(
    initial_position = 0.0,
    initial_velocity = 0.0,
    resistance = 1.0,
    mass = 1.0,
    gravity = 1.0,
    landscape_func= lambda x: 1.0 * x, ## upwards slope
    time_increment = 1.0
)

# env for test
env = OneDSimulationEnv(
        target_location=2.0,
        dynamics_model = dynamics_model
    )

## agent for test
agent = OracleMPCCEMAgent(
    target_location = 2.0,
    dynamics_model=copy.deepcopy(dynamics_model),
    num_lookahead_steps=5,
    num_rollouts=500,
    cem_iters = 10,
    cem_cutoff=0.5,
    initial_sampling_variance=100
)

def test_dynamics_model_independence():
    obs, _ = env.reset()
    _ = agent.act(state=obs)

    ## should be no change in dynamics model after rollout
    env_dynamics_model_position = env.dynamics_model.position
    env_dynamics_model_velocity = env.dynamics_model.velocity
    check_exact_equality(env_dynamics_model_position, obs[0])
    check_exact_equality(env_dynamics_model_velocity, obs[1])

