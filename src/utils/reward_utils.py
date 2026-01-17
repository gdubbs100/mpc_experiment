def calculate_distance_to_target(position: float, target: float) -> float:
    return abs(position - target)

def control_penalised_reward(absolute_distance_to_target: float, control_value: float) -> float:
    """
    distance to target is absolute value
    control value is between [-1, 1]
    """
    return absolute_distance_to_target**2 + control_value**2

def calculate_reward(position: float, target: float, control_value: float) -> float:
    absolute_distance_to_target = calculate_distance_to_target(position=position, target=target)
    return control_penalised_reward(
        absolute_distance_to_target=absolute_distance_to_target,
        control_value=control_value
    )