import numpy as np

def check_exact_equality(expected, actual):
    assert actual ==  expected, f"actual: {actual} != expected: {expected}; diff: {actual - expected}"

def check_approximate_equality(expected, actual, eps = 1.0e-10):
    assert np.abs(actual - expected) < eps, f"actual: {actual} != expected: {expected}; diff: {actual - expected}"