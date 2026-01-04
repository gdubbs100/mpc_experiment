import numpy as np

from environment.dynamics_models import DynamicsModel
from environment.vehicle import Vehicle
from utils.reward_utils import calculate_reward

class OracleMPCAgent:

    def __init__(
            self,
            target_location: float,
            dynamics_model: DynamicsModel,
            vehicle: Vehicle,
            num_lookahead_steps: int,
            num_rollouts: int
    ):
        self.target_location = target_location
        self.dynamics_model = dynamics_model
        self.vehicle = vehicle
        self.num_lookahead_steps = num_lookahead_steps
        self.num_rollouts = num_rollouts

    def rollout(
            self, 
            initial_position: float, 
            initial_velocity: float, 
            remaining_fuel: float,
            actions: np.ndarray[float]
        ) -> np.ndarray:
        rollout_reward = 0.0
        position, velocity = initial_position, initial_velocity
        self.vehicle.reset(fuel_mass = remaining_fuel)
        for a in actions:
            ## TODO: is there a better way to get time_increment?
            applied_force = self.vehicle.generate_force(u=a, time_increment=self.dynamics_model.time_increment)
            position, velocity = self.dynamics_model.step(
                applied_force = applied_force, 
                mass = self.vehicle.mass,
                current_position = position,
                current_velocity = velocity
            )
            # next_reward = np.abs(position - self.target_location)
            # absolute_distance_tocalculate_distance_to_target
            rollout_reward += -calculate_reward(
                position = position,
                target = self.target_location,
                control_value = a
            )
            
        return rollout_reward
    
    def run_mpc_iter(self, initial_state, actions):

        rewards = np.zeros(self.num_rollouts)
        for i in range(self.num_rollouts):
            rollout_reward = self.rollout(
                initial_position = initial_state[0],
                initial_velocity = initial_state[1],
                remaining_fuel = initial_state[2],
                actions = actions[i,:]
            )
            rewards[i] = rollout_reward
        return rewards

class OracleRandomShootingAgent(OracleMPCAgent):

    def __init__(
            self,
            *args, 
            **kwargs
    ):
        super().__init__(*args, **kwargs)
     
    def random_shooting(self, initial_state):
        actions = (
                np.random.normal(
                    loc=np.repeat(np.zeros(self.num_lookahead_steps), self.num_rollouts), 
                    scale=np.repeat(np.ones(self.num_lookahead_steps), self.num_rollouts)
                )
            ).reshape(self.num_rollouts, self.num_lookahead_steps)
        
        u = np.tanh(actions) # tanh squashing
        rewards = self.run_mpc_iter(
            initial_state = initial_state, 
            actions = u
        )
        best_trajectory = np.argmax(rewards)
        # take first action of best trajectory
        return u[best_trajectory, 0]
    
    def act(self, state):
        u = self.random_shooting(initial_state=state)
        return u

class OracleMPPIAgent(OracleMPCAgent):
    """
    MPC using model path predictive integral (?) method
    """

    def __init__(
            self, 
            learning_iters: int,
            temperature: float,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.learning_iters = learning_iters
        self.temperature = temperature
        self.u_nominal = np.zeros((self.num_lookahead_steps))

    def act(self, state):
        u = self.run_mppi(initial_state=state)
        return u
    
    def run_mppi(self, initial_state):
        ## Do we run this through several iters?
        u_new = self.u_nominal.copy()
        for _ in range(self.learning_iters):
            noise = np.random.normal(
                loc = 0.0, scale = 1.0, 
                size = (self.num_rollouts, self.num_lookahead_steps)
            )

            actions = np.tanh(u_new + noise)
            rewards = self.run_mpc_iter(
                initial_state=initial_state,
                actions = actions
            )
            weights = self.compute_mppi_weights(-rewards)

            u_new = u_new + np.sum(weights[:,None] * noise, axis = 0)
            ## chatgpt recommends this to keep values within -1, +1 
            ## is it necessary/ best way?
            u_new = np.tanh(u_new).squeeze() 
        
        ## after running iters, shift control sequence forward for warm start
        u_shifted = np.roll(u_new, -1)
        u_shifted[-1] = 0.0
        self.u_nominal = u_shifted
        return u_new[0]

    def compute_mppi_weights(self, rewards):
        worst_reward = rewards.min()
        weights = np.exp(-(rewards - worst_reward) / self.temperature)
        weights /= weights.sum() + 1.0e-6
        return weights


class OracleCEMAgent(OracleMPCAgent):

    def __init__(
            self, 
            learning_iters: int,
            cem_cutoff: float,
            initial_sampling_variance: float,
            deterministic_actions: bool = True,
            *args,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.learning_iters = learning_iters
        self.cem_cutoff = cem_cutoff
        self.initial_sampling_variance = initial_sampling_variance
        self.deterministic_actions = deterministic_actions


    def act(self, state):
        mean, std = self.run_cem(initial_state=state)
        if self.deterministic_actions:
            return np.tanh(mean)
        else:
            return np.tanh(np.random.normal(loc=mean, scale=std))

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
   
    def run_cem(self, initial_state):
        # re-initialise each time...?
        means = np.zeros(self.num_lookahead_steps)
        stds = np.ones_like(means)*self.initial_sampling_variance

        for _ in range(self.learning_iters):
            actions = (
                np.random.normal(
                    loc=np.repeat(means, self.num_rollouts), 
                    scale=np.repeat(stds, self.num_rollouts)
                )
            ).reshape(self.num_rollouts, self.num_lookahead_steps)
            u = np.tanh(actions) # apply tanh squashing
            rewards = self.run_mpc_iter(initial_state, u)
            elite_actions = self.select_elite_actions(
                actions = actions,
                rewards = rewards
            )

            means = elite_actions.mean(axis=0)
            stds = np.maximum(elite_actions.std(axis=0), 1e-6)
        return means[0], stds[0]
