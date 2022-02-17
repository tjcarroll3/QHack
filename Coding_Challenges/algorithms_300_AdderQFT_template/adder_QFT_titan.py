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



    print (2.**len(wires))
    phases = []
    sum = m + 2**(len(wires)-1)
    print(sum)

    for angle in range(2**len(wires)):

        phases.append(angle*np.pi/(2**len(wires)))
    for wire in wires[len(wires)-2:0:-1]:
        qml.ControlledPhaseShift(1.*(2 ** wire) * (phases[sum] - phases[2**(len(wires)-1)]), wires=[wire, len(wires)-1])





    #qml.RZ(2.*rotation, wires=wires[len(wires)-1])
    #qml.RZ(2. * rotation, wires=wires[1])

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
