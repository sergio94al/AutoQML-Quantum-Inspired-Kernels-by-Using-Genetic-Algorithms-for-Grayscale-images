import numpy as np
from numpy import pi as π
import circuit

def coding_bits(b):
    c = [b[n:n+7] for n,i in enumerate(b) if n%7==0]
    c_p=[]
    coding_0=[]
    for i in range(len(c)):
        for j in c[i]:
            c_p.append(str(j))
    np.asarray(c_p)
    c = [c_p[n:n+7] for n,i in enumerate(c_p) if n%7==0]
    for i in c:
        coding_0.append(''.join(i))
    return coding_0

class CircuitConversor:

    def __init__(self, nqubits, nparameters):
        self.gates = gates = {}
        for n, suffix in enumerate(['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1101','1110','1111','1100']):
            angle = (π /8)*(n+1)
            gates['000'+suffix] = (self.make_rx_(angle), 1.0)
            gates['001'+suffix] = (self.make_cx(), 3.0)
            gates['010'+suffix] = (self.make_id(), 0.0)
            gates['011'+suffix] = (self.make_rx(angle), 1.0)
            gates['100'+suffix] = (self.make_rz(angle), 1.0)
            gates['101'+suffix] = (self.make_ry_(angle), 1.0)
            gates['110'+suffix] = (self.make_rz_(angle), 1.0)
            gates['111'+suffix] = (self.make_ry(angle), 1.0)
        self.nqubits = nqubits
        self.nparameters = nparameters
        self.circuit = circuit.Circuit(nqubits)

    def __call__(self, coding_0, parameters):
        k = 0
        cost = 0
        state = self.circuit.zero_state()
        for ndx, z in enumerate(coding_bits(coding_0)):
            qubit = ndx % self.nqubits
            target = (ndx + 1) % self.nqubits
            fn, weight = self.gates[z]
            state, k = fn(state, parameters, k, qubit, target)
            cost += weight
        if k == 0 and parameters.ndim == 2:
            state = np.ones((parameters.shape[1],1)) * state.reshape(1,-1)
        return state, cost

    def make_id(self):
        def operation(state, parameters, k, qubit, target):
            return state, k
        return operation

    def make_h(self):
        def operation(state, parameters, k, qubit, target):
            return self.circuit.h(state, qubit), k
        return operation

    def make_cx(self):
        def operation(state, parameters, k, qubit, target):
            return self.circuit.cx(state, qubit, target), k
        return operation

    def make_rx(self, angle):
        def operation(state, parameters, k, qubit, target):
            ndx = k % self.nparameters
            return self.circuit.rx(state, parameters[ndx,:]*angle, qubit), k+1
        return operation

    def make_ry(self, angle):
        def operation(state, parameters, k, qubit, target):
            ndx = k % self.nparameters
            return self.circuit.ry(state, parameters[ndx,:]*angle, qubit), k+1
        return operation

    def make_rz(self, angle):
        def operation(state, parameters, k, qubit, target):
            ndx = k % self.nparameters
            return self.circuit.rz(state, parameters[ndx,:]*angle, qubit), k+1
        return operation
    
############################

    def make_rx_(self, angle):
        def operation(state, parameters, k, qubit, target):
            return self.circuit.rx(state,angle, qubit),k
        return operation

    def make_ry_(self, angle):
        def operation(state, parameters, k, qubit, target):
            return self.circuit.ry(state,angle, qubit),k
        return operation

    def make_rz_(self, angle):
        def operation(state, parameters, k, qubit, target):
            return self.circuit.rz(state,angle, qubit),k
        return operation
