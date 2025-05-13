# Net-QMPI
Net-QMPI is a Python package that provides a partial implementation of the Quantum Message Passing Interface (QMPI) and a full implementation of the Net Quantum Intermediate Representation (NetQIR) on top of the NetQASM SDK. NetQASM is a high-level framework designed to enable quantum network programming, supporting execution on simulators such as SimulaQron and NetSquid.

Like NetQIR, Net-QMPI aims to facilitate distributed quantum computing by enabling the execution of quantum programs across multiple nodes in a quantum network. This sets it apart from NetQASM’s original focus on programming individual quantum network nodes. Inspired by the classical MPI (Message Passing Interface) paradigm, Net-QMPI abstracts communication and follows a Single Program Multiple Data (SPMD) model adapted to quantum systems. This allows quantum programmers to develop distributed quantum applications in a style familiar from classical high-performance computing, making distributed quantum programming more accessible, modular, and portable.

## Example
### Send and receive

The repository includes a code example demonstrating a simple send/receive interaction between two quantum nodes, following a client-server style pattern. This example serves to highlight the differences between low-level NetQASM programming and the high-level abstraction provided by Net-QMPI.

The comparison is structured as follows:

- **NetQASM version:** implemented using two separate files, one per node (node0.py and node1.py), where the user must manually manage entanglement generation, classical communication, and teleportation logic.

- **Net-QMPI version:** implemented using a single file leveraging the NetQMPI API, which automatically handles low-level quantum networking operations (entanglement, measurement corrections, etc.) using a message-passing interface similar to classical MPI.

### NetQMPI version
```python
from netqasm.sdk import Qubit
from netqmpi.sdk.communicator import QMPICommunicator

def main(app_config=None, rank=0, size=1):
    COMM_WORLD = QMPICommunicator(rank, size, app_config)

    next_rank = COMM_WORLD.get_next_rank(rank)
    previous_rank = COMM_WORLD.get_prev_rank(rank)

    with COMM_WORLD.connection:
        if rank == 0:
            # Create a qubit |+> to teleport
            q = Qubit(COMM_WORLD.connection)
            q.H()

            COMM_WORLD.qsend(q, next_rank)
        else:
            qubit_recv = COMM_WORLD.qrecv(previous_rank)
            measurement = qubit_recv.measure()
            COMM_WORLD.connection.flush()
```
### NetQASM version
Node 0 (receiver):

```python
from netqasm.runtime.settings import Simulator, get_simulator
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state
from netqasm.sdk.toolbox.sim_states import get_fidelity, qubit_from, to_dm


def main(app_config=None):
    log_config = app_config.log_config

    # Create a socket to recv classical information
    socket = Socket("receiver", "sender", log_config=log_config)

    # Create a EPR socket for entanglement generation
    epr_socket = EPRSocket("sender")

    # Initialize the connection
    receiver = NetQASMConnection(
        app_name=app_config.app_name, log_config=log_config, epr_sockets=[epr_socket]
    )
    with receiver:
        epr = epr_socket.recv_keep()[0]
        receiver.flush()

        # Get the corrections
        m1, m2 = socket.recv_structured().payload
        if m2 == 1:
            epr.X()
        if m1 == 1:
            epr.Z()

        receiver.flush()
```

Node 1 (sender)
```python
from netqasm.logging.output import get_new_app_logger
from netqasm.runtime.settings import Simulator, get_simulator
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk.toolbox import set_qubit_state


def main(app_config=None, phi=0.0, theta=0.0):
    log_config = app_config.log_config
    app_logger = get_new_app_logger(app_name="sender", log_config=log_config)

    # Create a socket to send classical information
    socket = Socket("sender", "receiver", log_config=log_config)

    # Create a EPR socket for entanglement generation
    epr_socket = EPRSocket("receiver")

    # Initialize the connection to the backend
    sender = NetQASMConnection(
        app_name=app_config.app_name, log_config=log_config, epr_sockets=[epr_socket]
    )
    with sender:
        # Create a qubit to teleport
        q = Qubit(sender)
        set_qubit_state(q, phi, theta)

        # Create EPR pairs
        epr = epr_socket.create_keep()[0]

        # Teleport
        q.cnot(epr)
        q.H()
        m1 = q.measure()
        m2 = epr.measure()

    # Send the correction information
    m1, m2 = int(m1), int(m2)

    socket.send_structured(StructuredMessage("Corrections", (m1, m2)))

    return {"m1": m1, "m2": m2}
```

## Dependencies
- [NetQASM](https://netqasm.readthedocs.io/en/release-1.0/) >= 1.0.0