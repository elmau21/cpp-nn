#ifndef GRU_H
#define GRU_H

#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>

class GRU {
private:
    int input_size;   // Tamaño de entrada
    int hidden_size;  // Tamaño de la capa oculta
    std::vector<std::vector<double>> Wz, Wr, Wh; // Pesos
    std::vector<double> bz, br, bh; // Sesgos

public:
    GRU(int input_size, int hidden_size);
    std::vector<double> forward(const std::vector<double>& input, const std::vector<double>& prev_h);
};

#endif // GRU_H
