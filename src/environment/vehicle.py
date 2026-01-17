import torch
from utils.environment_utils import sign

class Vehicle:

    def __init__(
            self, 
            base_mass: float, 
            fuel_mass: float, 
            fuel_efficiency: float,
            fuel_burn_rate: float,
            finite_fuel: bool = False
        ):
        self.base_mass = base_mass # mass of body excluding fuel
        self.fuel_mass = fuel_mass # quantity of fuel measured in mass
        self.fuel_efficiency = fuel_efficiency # conversion rate of fuel to force
        self.fuel_burn_rate = fuel_burn_rate # how much fuel is burnt per second
        self.finite_fuel = finite_fuel # track fuel or keep unlimited
    
    @property
    def mass(self):
        return self.base_mass + self.fuel_mass
    
    def generate_force(self, u: float, time_increment: float) -> float:
        """
        u: normalized control in [-1, 1]
        """
        if abs(u) > 1.0:
            raise ValueError(f"Control value must be within +/- 1. Actual value: {u}")
        
        raw_fuel_burn = abs(u) * self.fuel_burn_rate

        # restrict amount of fuel to burn by available fuel
        fuel_burn = min(raw_fuel_burn, self.fuel_mass / time_increment)

        fuel_burned = fuel_burn * time_increment

        if self.finite_fuel:
            if isinstance(fuel_burned, torch.Tensor):
                fuel_burned = fuel_burned.detach()
            self.fuel_mass -= fuel_burned

        force = self.fuel_efficiency * fuel_burn * sign(u)

        return force
    
    def reset(self, fuel_mass: float) -> float:
        """
        Resets fuel mass
        """
        self.fuel_mass = fuel_mass