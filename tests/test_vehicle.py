import pytest
from environment.vehicle import Vehicle
from testing_utils import check_exact_equality

def test_fuel_tracking():
    v = Vehicle(
        base_mass = 1.0,
        fuel_mass = 1.0,
        fuel_efficiency = 1.0,
        finite_fuel=True
    )
    ## mass should be base_mass + fuel_mass before any useage
    check_exact_equality(expected=2.0, actual=v.mass)

    
    ## fuel_efficiency of 1.0 implies each unit of mass offers 1 unit of force
    force = v.generate_force(fuel_to_use = 1.0)
    check_exact_equality(expected=1.0, actual=force)

    ## mass should reduce to 1.0 after using all fuel
    check_exact_equality(expected=1.0, actual=v.mass)