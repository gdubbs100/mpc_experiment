import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algo.benchmark_agents import RandomAgent
from algo.mpc_agents import (
    OracleCEMAgent, 
    OracleRandomShootingAgent, 
    OracleMPPIAgent
)
from utils.logging_utils import (
    create_run_dir, 
    log_results_df, 
    log_video
)
from environment.dynamics_models import DynamicsModel
from environment.simulation_environment import OneDSimulationEnv
from environment.vehicle import Vehicle
from environment.viewer import MotionViewer
### create
TARGET_LOCATION = 5.0
VEHICLE = Vehicle(
    base_mass = 1.0,
    fuel_mass = 0.5,
    fuel_efficiency=300.0,
    fuel_burn_rate = 0.1,
    finite_fuel=True
)
DYNAMICS_MODEL = DynamicsModel(
    resistance=0.1,
    gravity=9.81,
    landscape_func=lambda x: np.sin(3*x) - x * np.cos(x),
    time_increment = 0.05
)

env = OneDSimulationEnv(
    target_location = TARGET_LOCATION,
    dynamics_model=DYNAMICS_MODEL,
    vehicle=VEHICLE,
    max_duration=100
)

# agent = RandomAgent()
ALGO = "CEM"
agent = OracleCEMAgent(
    target_location=TARGET_LOCATION,
    dynamics_model=copy.deepcopy(DYNAMICS_MODEL),
    vehicle=copy.deepcopy(VEHICLE),
    num_lookahead_steps=20,
    num_rollouts = 500,
    learning_iters = 5,
    cem_cutoff = 0.95,
    initial_sampling_variance=1
)
# ALGO="RandomShooting"
# agent = OracleRandomShootingAgent(
#     target_location=TARGET_LOCATION,
#     dynamics_model=copy.deepcopy(DYNAMICS_MODEL),
#     vehicle=copy.deepcopy(VEHICLE),
#     num_lookahead_steps=20,
#     num_rollouts = 500,
# )
# ALGO = "MPPI"
# agent = OracleMPPIAgent(
#     target_location=TARGET_LOCATION,
#     dynamics_model=copy.deepcopy(DYNAMICS_MODEL),
#     vehicle=copy.deepcopy(VEHICLE),
#     num_lookahead_steps=20,
#     num_rollouts = 500,
#     learning_iters=5,
#     temperature=1.0
# )


def main():
    run_dir = create_run_dir(base = "logs", algo = ALGO)
    obs, info = env.reset()
    done = False
    # positions, velocities, fuel, rewards, actions = [obs[0]],[obs[1]],[obs[2]],[None], [0]
    i=0
    results_dict = {
        i:{
            'positions': obs[0],
            'velocities': obs[1],
            'fuel': obs[2],
            'rewards': None,
            'actions':0
        }
    }
    
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        results_dict[i] = {
            'positions': next_obs[0],
            'velocities': next_obs[1],
            'fuel': next_obs[2],
            'rewards': reward,
            'actions':action
        }

        obs = next_obs
        i+=1

    log_results_df(run_dir=run_dir, results_dict = results_dict)

    video = MotionViewer(
        positions = np.array([results_dict[i]['positions'] for i in results_dict.keys()]),
        velocities= np.array([results_dict[i]['velocities'] for i in results_dict.keys()]),
        landscape_func = DYNAMICS_MODEL.landscape_func,
        target_x=  env.target_location,
        xlim=None,
    )
    log_video(run_dir=run_dir, viewer_class_object=video)




if __name__ == "__main__":
    main()
