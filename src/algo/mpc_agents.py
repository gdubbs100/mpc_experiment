import numpy as np
from typing import Callable

from environment.dynamics_models import DynamicsModel


class OracleMPCCEMAgent:
    """
    Takes a DynamicsModel as input and finds optimal sequence of actions using CEM
    Currently no constraints on the magnitude of actions
    """

    def __init__(
            self, 
            target_location: float, 
            dynamics_model: DynamicsModel,
            num_lookahead_steps: int, 
            num_rollouts: int, 
            cem_iters: int,
            cem_cutoff: float,
            initial_sampling_variance: float,
            deterministic_actions: bool = True
        ):

        self.target_location = target_location
        self.num_lookahead_steps = num_lookahead_steps
        self.num_rollouts = num_rollouts
        self.cem_iters = cem_iters
        self.cem_cutoff = cem_cutoff
        self.initial_sampling_variance = initial_sampling_variance
        self.deterministic_actions = deterministic_actions

        self.dynamics_model = dynamics_model

    def act(self, state):
        mean, std = self.run_cem_mpc(initial_state=state)
        if self.deterministic_actions:
            return mean
        else:
            return np.random.normal(loc=mean, scale=std)

    def select_elite_actions(self, 
                            actions: np.ndarray, 
                            rewards: np.ndarray, 
                            min_elite: int = 1):
        # actions: (N_rollouts, horizon), rewards: (N_rollouts,)
        N = actions.shape[0]
        k = max(min_elite, int(np.ceil(N * (1.0 - self.cem_cutoff))))   # e.g. cutoff=0.95 -> keep top 5%
        k = min(k, N)
        # indices of top-k rewards
        topk_idx = np.argsort(rewards)[-k:]
        elite = actions[topk_idx, :]
        return elite

    def rollout(
            self, 
            initial_position: float, 
            initial_velocity: float, 
            actions: np.ndarray[float]
        ) -> np.ndarray:
        rollout_reward = 0.0
        ## reset the dynamics model to the initial values
        self.dynamics_model.reset(
            initial_position=initial_position,
            initial_velocity=initial_velocity
        )
        for a in actions:
            # dynamics model tracks updates to position and velocity
            next_position, _ = self.dynamics_model.step(applied_force = a)
            next_reward = np.abs(next_position - self.target_location)
            rollout_reward += -next_reward
        return rollout_reward
    
    def run_cem_iter(self, initial_state, actions):

        rewards = np.zeros(self.num_rollouts)
        for i in range(self.num_rollouts):
            rollout_reward = self.rollout(
                initial_position = initial_state[0],
                initial_velocity = initial_state[1],
                actions = actions[i,:]
            )
            rewards[i] = rollout_reward
        return rewards
    
    def run_cem_mpc(self, initial_state):
        # re-initialise each time...?
        means = np.zeros(self.num_lookahead_steps)
        stds = np.ones_like(means)*self.initial_sampling_variance

        for _ in range(self.cem_iters):
            actions = (
                np.random.normal(
                    loc=np.repeat(means, self.num_rollouts), 
                    scale=np.repeat(stds, self.num_rollouts)
                )
            ).reshape(self.num_rollouts, self.num_lookahead_steps)
            rewards = self.run_cem_iter(initial_state, actions)
            # elite_actions = actions[rewards > np.quantile(rewards, self.cem_cutoff), :]
            elite_actions = self.select_elite_actions(
                actions = actions,
                rewards = rewards
            )

            means = elite_actions.mean(axis=0)
            stds = np.maximum(elite_actions.std(axis=0), 1e-6)
        return means[0], stds[0]
