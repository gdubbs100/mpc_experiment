import pytest
import numpy as np

from environment.dynamics_models import DynamicsModel
from testing_utils import (
    check_approximate_equality
)

## DynamicsModel to test
dynamics_model = DynamicsModel(
    resistance = 1.0,
    gravity = 1.0,
    landscape_func= lambda x: 1.0 * x, ## upwards slope
    time_increment = 1.0
)

def test_gravity_force_calculation():
    # gradient s = 1.0 (by def of landscape_func)
    # mass = 1.0
    # gravity = 1.0
    # force_from_gravity = -1.0 * 1.0 * 1 / sqrt(2) = - 1 / sqrt(2)
    actual = dynamics_model.calculate_force_from_gravity(
        current_position = 0.0,
        mass = 2.0,
    )
    expected = -np.sqrt(2)
    check_approximate_equality(
        expected=expected,
        actual=actual
    )

def test_acceleration_calculation():
    # force_from_gravity = -np.sqrt(2)
    # friction = 1.0 * 0 = 0
    # mass = 2.0
    # acceleration = (-np.sqrt(2) + 1.0 - 0) / 2.0 
    actual = dynamics_model.calculate_acceleration(
        applied_force = 1.0,
        current_velocity = 0.0,
        current_position = 0.0,
        mass = 2.0
    )
    expected = (-np.sqrt(2) + 1.0)/2.0
    check_approximate_equality(
        expected=expected,
        actual=actual
    )

def test_position_update():
    ## actual position / velocity
    # acceleration = 1 - 1/np.sqrt(2)
    # time increment is 1
    # velocity=0
    # so velocity=acceleration
    # position = position+aceleration
    current_position = 0.0
    current_velocity = 0.0
    acceleration = 1 - 1/np.sqrt(2)
    actual_position, actual_velocity = dynamics_model.update_position(
        applied_force = 1.0,
        mass = 1.0,
        current_position = current_position,
        current_velocity = current_velocity
    )
    expected_position = current_position + acceleration
    check_approximate_equality(
        expected=expected_position,
        actual=actual_position
    )
    expected_velocity = current_velocity + acceleration
    check_approximate_equality(
        expected=expected_velocity,
        actual=actual_velocity
    )