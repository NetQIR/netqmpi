from netqmpi.sdk.environment import Environment
from typing import Tuple
import numpy as np

n_qubits = 4

def theta(target: int, control: int) -> float:
    """
    Calculate the rotation angle for the controlled rotation gate.

    Args:
        target: The index of the target qubit.
        control: The index of the control qubit.
    Returns:
        The rotation angle in radians.
    """
    return - np.pi / (2 ** (target - control))

def theta_remote(target: Tuple[int, int], control: Tuple[int, int]) -> float:
    """
    Calculate the rotation angle for the controlled rotation gate between remote nodes.

    Args:
        target: A tuple containing the node index and the local index of the target qubit.
        control: A tuple containing the node index and the local index of the control qubit.
    Returns:
        The rotation angle in radians.
    """

    target_node, target_local = target
    control_node, control_local = control

    if target_node == control_node:
        return theta(target_local, control_local)
    else:
        return - np.pi / (2 ** (target_local - control_local + n_qubits * (target_node - control_node)))

def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank
    ROOT_RANK = 0

    print(f"[rank {rank}] main() started, comm.size={comm.size}")

    with comm:
        print(f"[rank {rank}] creating circuit")
        circuit = env.create_circuit(num_qubits=n_qubits + 1, num_clbits=n_qubits)

        for node in range(comm.size):
            print(f"[rank {rank}] node loop: node={node}")

            if rank == node:
                print(f"[rank {rank}] executing local node operations")

                circuit.h(0)

                for control in range(n_qubits - 1):
                    print(f"[rank {rank}] control={control}")

                    for target in range(control + 1, n_qubits):
                        print(f"[rank {rank}] crz local: control={control}, target={target}")
                        circuit.crz(theta(target, control), control, target)

                    for remote_node in range(rank + 1, comm.size):
                        print(f"[rank {rank}] qsend/qrecv control={control} to remote_node={remote_node}")
                        circuit.qsend([control], remote_node)
                        circuit.qrecv([control], remote_node)

                    circuit.h(control + 1)

            if rank > node:
                print(f"[rank {rank}] executing remote node operations for node={node}")

                for remote_qubit in range(n_qubits):
                    print(f"[rank {rank}] qrecv remote_qubit={remote_qubit} from node={node}")
                    circuit.qrecv([n_qubits], node)

                    for local_qubit in range(n_qubits):
                        angle = theta_remote((rank, local_qubit), (node, remote_qubit))
                        print(
                            f"[rank {rank}] crz remote: node={node}, remote_qubit={remote_qubit}, "
                            f"local_qubit={local_qubit}, angle={angle}"
                        )
                        circuit.crz(angle, n_qubits, local_qubit)

                    print(f"[rank {rank}] qsend back to node={node}")
                    circuit.qsend([n_qubits], node)

        print(f"[rank {rank}] measuring")
        for i in range(n_qubits):
            circuit.measure(i, i)

    results = comm.results

    if rank == comm.size - 1:
        print(f"Measurement results: {results}")
        