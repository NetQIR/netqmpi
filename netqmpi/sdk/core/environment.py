

class Environment:
    def __init__(self, create_circuit):
        self.circuits = []
        
        def create_and_save_circuit(num_qubits, num_clbits, rank):
            circuit = create_circuit(num_qubits, num_clbits, rank)
            self.circuits.append(circuit)
            return circuit
        
        self.create_circuit = create_and_save_circuit
        
        
    