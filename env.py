import numpy as np
import torch
from data_generator import data_generator

class AMTAEnvironment:
    def __init__(self, missiles, targets, alpha=0.5):
        self.missile_count = len(missiles)
        self.target_count = len(targets)
        self.missiles = missiles
        self.targets = targets
        self.decision_matrices = np.zeros((self.missile_count, self.target_count))
        self.current_missile = 0
        self.alpha = alpha

    def reset(self):
        self.decision_matrices = np.zeros((self.missile_count, self.target_count))
        self.current_missile = 0
        return self.decision_matrices.flatten()

    def observe_missile_state(self, missile_index):
        # You can return specific state information about the given missile index
        # For simplicity, I'm returning the decision matrix row for the missile.
        return self.decision_matrices[missile_index]

    def describe_target(self, missile_index, target_index):
        # You can return specific information about a target for a given missile
        # For simplicity, I'm returning a one-hot encoded vector for the target index.
        target_vector = np.zeros(self.target_count)
        target_vector[target_index] = 1
        return target_vector

    def calculate_combat_effectiveness(self, matrix):
        return E(matrix)

    def step(self, action):
        target = action
        prev_state = np.copy(self.decision_matrices)
        self.decision_matrices[self.current_missile, target] = 1

        reward = reward_function(action, prev_state, self.alpha)

        self.current_missile += 1

        done = True if self.current_missile == self.missile_count else False

        return self.decision_matrices.flatten(), reward, done

def reward_function(a, X, alpha):
    X1 = state_translate(a, X)

    rl = E(X1) - E(X)

    assigned_missile_num = 1

    rg = E(X1)/assigned_missile_num

    r = alpha * rl + (1 - alpha) * rg

    return r

def state_translate(a, X):
    # This is a placeholder. You may need to implement the correct translation here.
    return X

def E(matrix):
    M = len(matrix)
    N = len(matrix[0]) if M > 0 else 0
    sum_value = 0
    for i in range(M):
        for j in range(N):
            if matrix[i][j] == 1:
                pij = missiles[i].dp / (missiles[i].dp + targets[j].cost)
                sum_value += pij * targets[j].value
    return sum_value

missile_number = int(3.5 * 10)
target_number = 10
missiles, targets = data_generator(missile_number, target_number)

env = AMTAEnvironment(missiles, targets)

