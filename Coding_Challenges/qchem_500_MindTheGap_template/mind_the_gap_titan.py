import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    qubits = 4

    dev = qml.device("default.qubit", wires=qubits)
    def circuit(param):
        # qml.PauliX(wires=0)
        # qml.PauliX(wires=1)
        qml.BasisState(np.array([1, 1, 0, 0]), wires=[0, 1, 2, 3])
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param)
        return qml.expval(H)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    energy = [cost_fn(theta)]

    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        energy.append(cost_fn(theta))
        angle.append(theta)

        conv = np.abs(energy[-1] - prev_energy)

        # if n % 2 == 0:
        #     print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break


    devstate = qml.device('default.qubit', wires=qubits)
    @qml.qnode(devstate)
    def ground_state(param):
        circuit(param)
        return qml.state()

    gState = ground_state(angle[-1]).real



    return energy[-1], gState
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #

    for i in range(len(ground_state)):
        ground_state[i] = ground_state[i] * beta


    projector = np.kron(ground_state, ground_state).reshape(16, 16)

    H0 = qml.utils.sparse_hamiltonian(H).real
    H0_matrix = H0.toarray()

    H1_matrix = projector + H0_matrix

    # ket = ground_state.reshape(4, 1)
    # bra = ground_state.reshape(1, 4)
    # projector = np.multiply(ket, bra)
    # projector = np.multiply(beta, projector)
    # ob2 = qml.Hermitian(projector, wires=[0, 1])

    # H1 = H #.matrix[0] + projector
    # print(H.terms[0])

    #H1 = H.matrix + projector
    return qml.Hermitian(H1_matrix, wires=range(4))
    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #
    qubits = 4

    dev = qml.device("default.qubit", wires=qubits)

    def circuit(param):
        # qml.PauliX(wires=0)
        # qml.PauliX(wires=1)
        qml.BasisState(np.array([1, 0, 1, 0]), wires=[0, 1, 2, 3]) # First excited state
        qml.DoubleExcitation(param, wires=[0, 1, 2, 3])

    @qml.qnode(dev)
    def cost_fn(param):
        circuit(param)
        return qml.expval(H1)

    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    theta = np.array(0.0, requires_grad=True)
    energy = [cost_fn(theta)]

    angle = [theta]

    max_iterations = 100
    conv_tol = 1e-06

    for n in range(max_iterations):
        theta, prev_energy = opt.step_and_cost(cost_fn, theta)

        loss = cost_fn(theta)
        angle.append(theta)

        conv = np.abs(loss - prev_energy)

        # if n % 2 == 0:
        #     print(f"Step = {n},  Energy = {energy[-1]:.8f} Ha")

        if conv <= conv_tol:
            break

    return cost_fn(theta)
    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)


    beta = 15.0
    H1 = create_H1(ground_state, beta, H)

    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
