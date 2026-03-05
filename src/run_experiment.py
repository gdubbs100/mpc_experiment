import argparse
import copy
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from algo.mpc_agents import (
    OracleCEMAgent, 
    OracleRandomShootingAgent, 
    OracleMPPIAgent,
    OracleGBAgent
)
from utils.logging_utils import (
    create_run_dir, 
    create_results_df,
    log_results_df, 
    log_video
)
from environment.dynamics_models import DynamicsModel
from environment.simulation_environment import OneDSimulationEnv
from environment.vehicle import Vehicle
from environment.viewer import MotionViewer

## set seed
torch.manual_seed(42)
np.random.seed(seed=42)

## Argparser
parser = argparse.ArgumentParser()
parser.add_argument('--experiment_id', type=str, help="name the experiment", required=True)
parser.add_argument('--num_runs', type = int, default=5, help="Number of runs for each algo to do", required=False)
parser.add_argument('--num_lookahead_steps', type = int, default=50, help="number of lookahead steps for MPC")
parser.add_argument('--use_terminal_reward', type = bool, default = True, help = "use a terminal reward")
args = parser.parse_args()

## experiment values
EXPERIMENT_ID = args.experiment_id
NUM_RUNS = args.num_runs
TARGET_LOCATION = 5.0
def landscape_func(x):
    if isinstance(x, torch.Tensor):
        return torch.sin(3*x) - x * torch.cos(x)
    else:
        return np.sin(3*x) - x * np.cos(x)

LANDSCAPE_FUNC = lambda x: landscape_func(x)
NUM_LOOKAHEAD_STEPS = args.num_lookahead_steps
NUM_ROLLOUTS = 500
LEARNING_ITERS = 5
USE_TERMINAL_REWARD = args.use_terminal_reward

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
    landscape_func=LANDSCAPE_FUNC,
    time_increment = 0.05
)

env = OneDSimulationEnv(
    target_location = TARGET_LOCATION,
    dynamics_model=DYNAMICS_MODEL,
    vehicle=VEHICLE,
    max_duration=100
)

ALGOS = {
    "CEM":OracleCEMAgent(
        target_location=TARGET_LOCATION,
        dynamics_model=copy.deepcopy(DYNAMICS_MODEL),
        vehicle=copy.deepcopy(VEHICLE),
        num_lookahead_steps=NUM_LOOKAHEAD_STEPS,
        num_rollouts = NUM_ROLLOUTS,
        use_terminal_reward = USE_TERMINAL_REWARD,
        learning_iters = LEARNING_ITERS,
        cem_cutoff = 0.95,
        initial_sampling_variance=1
    ),
    "RandomShooting": OracleRandomShootingAgent(
        target_location=TARGET_LOCATION,
        dynamics_model=copy.deepcopy(DYNAMICS_MODEL),
        vehicle=copy.deepcopy(VEHICLE),
        num_lookahead_steps=NUM_LOOKAHEAD_STEPS,
        num_rollouts = NUM_ROLLOUTS,
        use_terminal_reward = USE_TERMINAL_REWARD,
    ),
    "MPPI": OracleMPPIAgent(
        target_location=TARGET_LOCATION,
        dynamics_model=copy.deepcopy(DYNAMICS_MODEL),
        vehicle=copy.deepcopy(VEHICLE),
        num_lookahead_steps=NUM_LOOKAHEAD_STEPS,
        num_rollouts = NUM_ROLLOUTS,
        use_terminal_reward = USE_TERMINAL_REWARD,
        learning_iters = LEARNING_ITERS,
        temperature=1.0
    ),
    "GB": OracleGBAgent(
        target_location=TARGET_LOCATION,
        dynamics_model=copy.deepcopy(DYNAMICS_MODEL),
        vehicle=copy.deepcopy(VEHICLE),
        num_lookahead_steps=NUM_LOOKAHEAD_STEPS,
        use_terminal_reward = USE_TERMINAL_REWARD,
        num_rollouts = 1,
        learning_iters = 50,
        learning_rate=0.003
    )

}

def main():
    for algo, agent in ALGOS.items():
        run_dir = create_run_dir(base = f"logs/{EXPERIMENT_ID}", algo = algo)
        all_results = pd.DataFrame()
        for j in range(NUM_RUNS):
            print(f"Running {algo} run {j} for experiment: {EXPERIMENT_ID}...")
            obs, info = env.reset()
            done = False
            i=0
            results_dict = {
                i:{
                    'positions': obs[0],
                    'velocities': obs[1],
                    'fuel': obs[2],
                    'rewards': None,
                    'actions':0,
                    'distance_to_target': info['distance_to_target'],
                    'run': j
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
                    'actions':action,
                    'distance_to_target': info['distance_to_target'],
                    'run': j
                }

                obs = next_obs
                i+=1
            tmp_results_df = create_results_df(results_dict=results_dict)
            all_results = pd.concat([tmp_results_df, all_results])

            video = MotionViewer(
                positions = np.array([results_dict[i]['positions'] for i in results_dict.keys()]),
                velocities= np.array([results_dict[i]['velocities'] for i in results_dict.keys()]),
                landscape_func = DYNAMICS_MODEL.landscape_func,
                target_x = env.target_location,
                xlim=None,
            )
            log_video(run_dir=run_dir, viewer_class_object=video, id=j)
        print(f"logging results for {algo}...")
        log_results_df(run_dir = run_dir, results_df=all_results)







if __name__ == "__main__":
    main()
