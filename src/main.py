import numpy as np
import matplotlib.pyplot as plt

from algo.benchmark_agents import RandomAgent

from environment.dynamics_models import DynamicsModel
from environment.simulation_environment import OneDSimulationEnv
from environment.viewer import MotionViewer

dynamics_model = DynamicsModel(
    initial_position=0.0,
    initial_velocity=0.0,
    resistance=0.1,
    mass = 1.0,
    gravity=9.81,
    landscape_func=lambda x: np.sin(x),
    time_increment = 0.05
)

env = OneDSimulationEnv(
    target_location = 2.0,
    dynamics_model=dynamics_model,
)

agent = RandomAgent()


def main():
    obs, info = env.reset()
    done = False
    positions, velocities, rewards, actions = [obs[0]],[obs[1]],[],[]
    while not done:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        positions.append(next_obs[0])
        velocities.append(next_obs[1])
        rewards.append(reward)
        actions.append(action)
        obs = next_obs
    
    # fig, ax = plt.subplots(2,2,figsize = (10, 7))
    # ax = ax.flatten()
    # ax[0].plot(positions, label='positions')
    # ax[0].set_title('positions')
    # ax[1].plot(velocities, label='velocities')
    # ax[1].set_title('velocities')
    # ax[2].plot(rewards, label='rewards')
    # ax[2].set_title('rewards')
    # ax[3].plot(actions, label='actions')
    # ax[3].set_title('actions')
    
    # plt.show()

    video = MotionViewer(
        positions = np.array(positions),
        velocities= np.array(velocities),
        landscape_func = dynamics_model.landscape_func,
        target_x=  env.target_location,
        xlim=None,
    )

    video.save("../video_logs/episode.mp4")



if __name__ == "__main__":
    main()
