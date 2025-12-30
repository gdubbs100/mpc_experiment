import numpy as np
from typing import Any

class RandomAgent:

    def act(self, state):
        return np.random.normal(loc=0, scale=1.0, size = 1).item()
    
class OneAndDoneAgent:

    def __init__(self, done_time: int = 1, initial_action: float = 10.0):
        self.done_time = done_time
        self.initial_action = initial_action
        self.t = 0

    def act(self, state: Any) -> float:
        if self.t < self.done_time:
            self.t += 1
            return self.initial_action
        else:
            return 0.0