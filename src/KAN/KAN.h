#ifndef KAN_H
#define KAN_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

class KAN {
private:
    int input_size;   // Tamaño de entrada
    int hidden_size;  // Tamaño de la capa oculta
    int output_size;  // Tamaño de salida
    std::vector<std::vector<double>> weights_input_hidden; // Pesos de entrada a capa oculta
    std::vector<double> weights_hidden_output; // Pesos de capa oculta a salida

    // Función de activación ReLU
    double relu(double x) {
        return std::max(0.0, x);
    }

public:
    // Constructor de la KAN
    KAN(int input_size, int hidden_size, int output_size);

    // Método para realizar la propagación hacia adelante
    std::vector<double> forward(const std::vector<double>& input);
};

#endif // KAN_H
