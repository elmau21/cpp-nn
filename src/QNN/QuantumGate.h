#ifndef QUANTUM_GATE_H
#define QUANTUM_GATE_H

#include <complex>
#include <vector>
#include <cmath>
#include <iostream>

using Complex = std::complex<double>;
using Matrix = std::vector<std::vector<Complex>>;

class QuantumGate {
public:
    // Hadamard Gate
    static Matrix hadamard() {
        return {
            {Complex(1 / sqrt(2)), Complex(1 / sqrt(2))},
            {Complex(1 / sqrt(2)), Complex(-1 / sqrt(2))}
        };
    }

    // Pauli-X Gate (Quantum NOT gate)
    static Matrix pauliX() {
        return {
            {Complex(0), Complex(1)},
            {Complex(1), Complex(0)}
        };
    }

    // Pauli-Z Gate (Phase flip gate)
    static Matrix pauliZ() {
        return {
            {Complex(1), Complex(0)},
            {Complex(0), Complex(-1)}
        };
    }

    // Identity Gate
    static Matrix identity() {
        return {
            {Complex(1), Complex(0)},
            {Complex(0), Complex(1)}
        };
    }

    // Function to apply a quantum gate to a qubit
    static std::vector<Complex> applyGate(const Matrix& gate, const std::vector<Complex>& qubit) {
        std::vector<Complex> result(2, Complex(0, 0));
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                result[i] += gate[i][j] * qubit[j];
            }
        }
        return result;
    }
};

#endif // QUANTUM_GATE_H
