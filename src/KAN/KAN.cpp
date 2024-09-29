#include "KAN.h"

// Constructor de la KAN
KAN::KAN(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
    // Inicializar pesos aleatoriamente
    weights_input_hidden.resize(hidden_size, std::vector<double>(input_size));
    weights_hidden_output.resize(hidden_size);

    // Asignar valores aleatorios a los pesos
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights_input_hidden[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
        weights_hidden_output[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

// Propagación hacia adelante
std::vector<double> KAN::forward(const std::vector<double>& input) {
    std::vector<double> hidden(hidden_size);
    std::vector<double> output(output_size);

    // Capa oculta
    for (int i = 0; i < hidden_size; ++i) {
        hidden[i] = 0;
        for (int j = 0; j < input_size; ++j) {
            hidden[i] += weights_input_hidden[i][j] * input[j];
        }
        hidden[i] = relu(hidden[i]); // Aplicar función de activación
    }

    // Capa de salida
    for (int i = 0; i < output_size; ++i) {
        output[i] = 0;
        for (int j = 0; j < hidden_size; ++j) {
            output[i] += hidden[j] * weights_hidden_output[j];
        }
    }

    return output;
}
