import random

class Target:
    def __init__(self, type_, value, cost):
        self.type = type_
        self.value = value
        self.cost = cost

    def __repr__(self):
        return f"Target({self.type}, {self.value}, {self.cost})"

class Missile:
    def __init__(self, type_, cost, dp, penetration_value):
        self.type = type_
        self.cost = cost
        self.dp = dp  # Damage potential
        self.penetration_value = penetration_value

    def __repr__(self):
        return f"Missile({self.type}, {self.cost}, {self.dp}, {self.penetration_value})"

def data_generator(missile_number, target_number):
    # Define the attributes of targets
    T_attributes = {
        "T1": Target("T1", 4, 1),
        "T2": Target("T2", 6, 2),
        "T3": Target("T3", 8, 4),
        "T4": Target("T4", 16, 8)
    }

    # Define the attributes of missiles
    M_attributes = {
        "M1": Missile("M1", 1, 6, 0.35),
        "M2": Missile("M2", 1.25, 6, 0.7)
    }

    # Generation of Targets
    targets = []
    targets.extend([T_attributes["T1"]] * (target_number // 2))
    targets.append(T_attributes["T4"])
    T2_count = random.randint(0, target_number // 2 - 1)
    targets.extend([T_attributes["T2"]] * T2_count)
    targets.extend([T_attributes["T3"]] * (target_number // 2 - T2_count - 1))

    # Generation of Missiles
    missiles = []
    missiles.extend([M_attributes["M1"]] * (2 * missile_number // 3))
    missiles.extend([M_attributes["M2"]] * (missile_number // 3))

    return missiles, targets


# Example usage
missile_number = int(3.5 * 10)  # Example with N=10 and ρ=3.5
target_number = 10
missiles, targets = data_generator(missile_number, target_number)
print("Missiles:", missiles)
print("Targets:", targets)
