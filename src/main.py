import copy

import numpy as np
import matplotlib.pyplot as plt

from algo.benchmark_agents import RandomAgent
from algo.mpc_agents import OracleMPCCEMAgent

from environment.dynamics_models import DynamicsModel
from environment.simulation_environment import OneDSimulationEnv
from environment.vehicle import Vehicle
from environment.viewer import MotionViewer

TARGET_LOCATION = 2.0
VEHICLE = Vehicle(
    base_mass = 1.0,
    fuel_mass = 0.2,
    fuel_efficiency=300.0,
    fuel_burn_rate = 0.1,
    finite_fuel=True
)
DYNAMICS_MODEL = DynamicsModel(
    # initial_position=0.0,
    # initial_velocity=0.0,
    resistance=0.1,
    gravity=9.81,
    landscape_func=lambda x: -np.sin(x/2) - x/2 * np.cos(x-1),
    time_increment = 0.05
)

env = OneDSimulationEnv(
    target_location = TARGET_LOCATION,
    dynamics_model=DYNAMICS_MODEL,
    vehicle=VEHICLE,
    max_duration=100
)

# agent = RandomAgent()
agent = OracleMPCCEMAgent(
    target_location=TARGET_LOCATION,
    dynamics_model=copy.deepcopy(DYNAMICS_MODEL),
    vehicle=copy.deepcopy(VEHICLE),
    num_lookahead_steps=30,
    num_rollouts = 500,
    cem_iters = 15,
    cem_cutoff = 0.95,
    initial_sampling_variance=1
)


def main():
    obs, info = env.reset()
    done = False
    positions, velocities, fuel, rewards, actions = [obs[0]],[obs[1]],[obs[2]],[], []
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        positions.append(next_obs[0])
        velocities.append(next_obs[1])
        fuel.append(next_obs[2])
        rewards.append(reward)
        actions.append(action)
        obs = next_obs
    
    fig, ax = plt.subplots(2,3,figsize = (10, 7))
    ax = ax.flatten()
    ax[0].plot(positions, label='positions')
    ax[0].set_title('positions')
    ax[1].plot(velocities, label='velocities')
    ax[1].set_title('velocities')
    ax[2].plot(rewards, label='rewards')
    ax[2].set_title('rewards')
    ax[3].plot(actions, label='actions')
    ax[3].set_title('actions')
    ax[4].plot(fuel, label='fuel')
    ax[4].set_title('fuel')
    
    plt.show()

    video = MotionViewer(
        positions = np.array(positions),
        velocities= np.array(velocities),
        landscape_func = DYNAMICS_MODEL.landscape_func,
        target_x=  env.target_location,
        xlim=None,
    )

    video.save("../video_logs/episode2.mp4")



if __name__ == "__main__":
    main()
