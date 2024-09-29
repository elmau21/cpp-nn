#ifndef LSTM_H
#define LSTM_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

class LSTM {
private:
    int input_size;   // Tamaño de entrada
    int hidden_size;  // Tamaño de la capa oculta
    std::vector<std::vector<double>> Wf, Wi, Wc, Wo; // Pesos
    std::vector<double> bf, bi, bc, bo; // Sesgos

public:
    LSTM(int input_size, int hidden_size);
    std::vector<double> forward(const std::vector<double>& input, const std::vector<double>& prev_h, const std::vector<double>& prev_c);
};

#endif // LSTM_H
