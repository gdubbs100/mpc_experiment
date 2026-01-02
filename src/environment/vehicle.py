import numpy as np

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

    def track_fuel_use(self, u: float, time_increment: float) -> float:
        """
        u: normalized control in [-1, 1]
        returns: fuel burned
        """
        u = max(min(u, 1.0), -1.0)

        fuel_burn = self.fuel_burn_rate * abs(u) * time_increment
        fuel_burn = min(self.fuel_mass, fuel_burn)

        self.fuel_mass -= fuel_burn

    def generate_force(self, u: float, time_increment: float) -> float:
        """
        u: normalized control in [-1, 1]
        """
        if abs(u) > 1.0:
            raise ValueError(f"Control value must be within +/- 1. Actual value: {u}")

        if self.finite_fuel:
            self.track_fuel_use(u, time_increment)

        # Force is *entirely* determined by fuel physics
        force = self.fuel_burn_rate * self.fuel_efficiency * u

        return force
    
    def reset(self, fuel_mass: float) -> float:
        """
        Resets fuel mass
        """
        self.fuel_mass = fuel_mass