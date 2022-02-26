#! /usr/bin/python3

import sys
import pennylane as qml
from pennylane import numpy as np


dev = qml.device("default.qubit", wires=2)


def prepare_entangled(alpha, beta):
    """Construct a circuit that prepares the (not necessarily maximally) entangled state in terms of alpha and beta
    Do not forget to normalize.

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>
    """

    # QHACK #
    amplitudes = [alpha, 0, 0, beta]
    qml.AmplitudeEmbedding(amplitudes, wires=[0, 1], normalize=True)  # need to normalize this to avoid norm error

    # QHACK #

@qml.qnode(dev)
def chsh_circuit(theta_A0, theta_A1, theta_B0, theta_B1, x, y, alpha, beta):
    """Construct a circuit that implements Alice's and Bob's measurements in the rotated bases

    Args:
        - theta_A0 (float): angle that Alice chooses when she receives x=0
        - theta_A1 (float): angle that Alice chooses when she receives x=1
        - theta_B0 (float): angle that Bob chooses when he receives x=0
        - theta_B1 (float): angle that Bob chooses when he receives x=1
        - x (int): bit received by Alice
        - y (int): bit received by Bob
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (np.tensor): Probabilities of each basis state
    """

    prepare_entangled(alpha, beta)

    # QHACK #
    def rotate(angle, wire):
        qml.RY(-2.*angle, wires=wire)

    if x == 0:
        rotate(theta_A0, 0)
    else:
        rotate(theta_A1, 0)

    if y == 0:
        rotate(theta_B0, 1)
    else:
        rotate(theta_B1, 1)

    # QHACK #

    return qml.probs(wires=[0, 1])
    

def winning_prob(params, alpha, beta):
    """Define a function that returns the probability of Alice and Bob winning the game.

    Args:
        - params (list(float)): List containing [theta_A0,theta_A1,theta_B0,theta_B1]
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning the game
    """

    def getProbs(x, y):
        return chsh_circuit(params[0], params[1], params[2], params[3], x, y, alpha, beta)

    # QHACK #
    probs = getProbs(0, 0)
    win00 = probs[0] + probs[3]

    probs = getProbs(0, 1)
    win01 = probs[0] + probs[3]

    probs = getProbs(1, 0)
    win10 = probs[0] + probs[3]

    probs = getProbs(1, 1)
    win11 = probs[1] + probs[2]

    avg_prob = (win00 + win01 + win10 + win11)/4.
    return avg_prob
    # QHACK #
    

def optimize(alpha, beta):
    """Define a function that optimizes theta_A0, theta_A1, theta_B0, theta_B1 to maximize the probability of winning the game

    Args:
        - alpha (float): real coefficient of |00>
        - beta (float): real coefficient of |11>

    Returns:
        - (float): Probability of winning
    """

    def cost(params):
        """Define a cost function that only depends on params, given alpha and beta fixed"""
        return (1. - winning_prob(params, alpha, beta))**2
    # QHACK #

    #Initialize parameters, choose an optimization method and number of steps
    init_params = np.array([np.pi/4, -np.pi/4, np.pi/4, 0], requires_grad = True) # np.array([0, np.pi/4, -np.pi/4, 0.1], requires_grad = True)
    opt = qml.GradientDescentOptimizer(0.3)
    steps = 200

    # QHACK #
    
    # set the initial parameter values
    params = init_params

    for i in range(steps):
        # update the circuit parameters 
        # QHACK #

        params = opt.step(cost, params)

        # QHACK #

    return winning_prob(params, alpha, beta)


if __name__ == '__main__':
    inputs = sys.stdin.read().split(",")
    output = optimize(float(inputs[0]), float(inputs[1]))
    print(f"{output}")