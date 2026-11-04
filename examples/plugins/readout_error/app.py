"""
Measuring known states, so that readout error is all that varies.

Every rank measures a qubit in ``|0⟩`` and a qubit in ``|1⟩``. Without noise
each rank reads ``10`` (qubit 1 first) on every shot; any other outcome is a
readout error, so the counts show ``prob_error_0`` and ``prob_error_1``
directly.

On the NetQASM backend only the last rank receives its counts inside
``main``, so that is the rank that reports them.

Run from this directory, so that the plugin can be imported::

    netqmpi -n 2 app.py --netqasm --config config.yaml

and compare with the same program without the plugin::

    netqmpi -n 2 app.py --netqasm
"""
from netqmpi.sdk.environment import Environment


def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=2, num_clbits=2)
        circuit.x(1)                        # qubit 0 stays |0>, qubit 1 is |1>
        circuit.measure(0, 0)
        circuit.measure(1, 1)

    if comm.results:
        counts = comm.results
        shots = sum(counts.values())
        # Keys read qubit 1 first: "10" is the outcome without error.
        misread_0 = sum(n for key, n in counts.items() if key[1] == "1") / shots
        misread_1 = sum(n for key, n in counts.items() if key[0] == "0") / shots
        print(f"rank {comm.rank}: {counts}")
        print(f"  |0> read as 1 in {misread_0:.1%} of {shots} shots")
        print(f"  |1> read as 0 in {misread_1:.1%} of {shots} shots")
        # What the plugin itself counted, every rank's measurements included.
        plugin = comm.metrics.get("plugin")
        if plugin:
            print(f"  plugin: {plugin['flipped']} of {plugin['measurements']} "
                  f"measurements misread")
