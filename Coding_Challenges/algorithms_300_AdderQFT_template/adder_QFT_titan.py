#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qfunc_adder(m, wires):
    """Quantum function capable of adding m units to a basic state given as input.

    Args:
        - m (int): units to add.
        - wires (list(int)): list of wires in which the function will be executed on.
    """

    qml.QFT(wires=wires)

    # QHACK #

    N = len(wires)
    n = 2 ** (N - 1)
    sum = n + m

    phaseN = []
    for i in range(2 ** N):
        phaseN.append(i * 2. * np.pi / (2. ** N))

    for wire in wires[len(wires) - 1::-1]:
        phase_init = 2. ** (N - 1 - wire) * phaseN[n]
        phase_final = 2. ** (N - 1 - wire) * phaseN[sum]
        delta_phase = phase_final - phase_init

        qml.PhaseShift(delta_phase, wires=wire)


    # QHACK #

    qml.QFT(wires=wires).inv()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    m = int(inputs[0])
    n_wires = int(inputs[1])
    wires = range(n_wires)

    dev = qml.device("default.qubit", wires=wires, shots=1)

    @qml.qnode(dev)
    def test_circuit():
        # Input:  |2^{N-1}>
        qml.PauliX(wires=0)

        qfunc_adder(m, wires)
        return qml.sample()

    output = test_circuit()
    print(*output, sep=",")
