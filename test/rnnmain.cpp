#include <iostream>
#include "../src/RNN/LSTM/LSTM.h"
#include "../src/RNN/GRU/GRU.h"

int main() {
    // Tamaño de entrada y oculta
    int input_size = 3;
    int hidden_size = 5;

    // Crear instancias de LSTM y GRU
    LSTM lstm(input_size, hidden_size);
    GRU gru(input_size, hidden_size);

    // Entradas de prueba
    std::vector<double> input = {1.0, 0.5, -1.5};
    std::vector<double> prev_h(hidden_size, 0.0); // Estado oculto anterior inicial
    std::vector<double> prev_c(hidden_size, 0.0); // Estado de celda anterior inicial

    // Propagación hacia adelante en LSTM
    std::vector<double> lstm_output = lstm.forward(input, prev_h, prev_c);
    std::cout << "Salida de LSTM: ";
    for (const auto& val : lstm_output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Propagación hacia adelante en GRU
    std::vector<double> gru_output = gru.forward(input, prev_h);
    std::cout << "Salida de GRU: ";
    for (const auto& val : gru_output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}