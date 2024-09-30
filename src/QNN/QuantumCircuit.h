#ifndef QUANTUM_CIRCUIT_H
#define QUANTUM_CIRCUIT_H

#include "QuantumGate.h"
#include <vector>
#include <cmath>
#include <iostream>

class QuantumCircuit {
private:
    std::vector<Complex> qubit;

public:
    QuantumCircuit() {
        // Initialize the qubit in the |0⟩ state
        qubit = {Complex(1, 0), Complex(0, 0)};
    }

    void applyHadamard() {
        qubit = QuantumGate::applyGate(QuantumGate::hadamard(), qubit);
    }

    void applyPauliX() {
        qubit = QuantumGate::applyGate(QuantumGate::pauliX(), qubit);
    }

    void applyPauliZ() {
        qubit = QuantumGate::applyGate(QuantumGate::pauliZ(), qubit);
    }

    std::vector<Complex> getQubitState() const {
        return qubit;
    }

    void measure() {
        double probability0 = std::norm(qubit[0]);
        double probability1 = std::norm(qubit[1]);
        
        std::cout << "Measurement probabilities:" << std::endl;
        std::cout << "|0⟩: " << probability0 << std::endl;
        std::cout << "|1⟩: " << probability1 << std::endl;

        // Measurement simulation
        double random = static_cast<double>(rand()) / RAND_MAX;
        if (random < probability0) {
            std::cout << "Measured: |0⟩" << std::endl;
        } else {
            std::cout << "Measured: |1⟩" << std::endl;
        }
    }
};

#endif // QUANTUM_CIRCUIT_H
