#include "GRU.h"
#include <numeric> // Agregar esta línea

// Constructor de GRU
GRU::GRU(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {
    // Inicializar pesos y sesgos aleatorios
    Wz.resize(hidden_size, std::vector<double>(input_size + hidden_size));
    Wr.resize(hidden_size, std::vector<double>(input_size + hidden_size));
    Wh.resize(hidden_size, std::vector<double>(input_size + hidden_size));

    bz.resize(hidden_size);
    br.resize(hidden_size);
    bh.resize(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size + hidden_size; ++j) {
            Wz[i][j] = static_cast<double>(rand()) / RAND_MAX;
            Wr[i][j] = static_cast<double>(rand()) / RAND_MAX;
            Wh[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
        bz[i] = static_cast<double>(rand()) / RAND_MAX;
        br[i] = static_cast<double>(rand()) / RAND_MAX;
        bh[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

// Función de activación sigmoide
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Propagación hacia adelante
std::vector<double> GRU::forward(const std::vector<double>& input, const std::vector<double>& prev_h) {
    std::vector<double> combined(input.size() + prev_h.size());
    std::copy(input.begin(), input.end(), combined.begin());
    std::copy(prev_h.begin(), prev_h.end(), combined.begin() + input.size());

    // Puertas
    std::vector<double> zt(hidden_size), rt(hidden_size), ht_hat(hidden_size);
    
    for (int i = 0; i < hidden_size; ++i) {
        zt[i] = sigmoid(std::inner_product(Wz[i].begin(), Wz[i].end(), combined.begin(), bz[i]));
        rt[i] = sigmoid(std::inner_product(Wr[i].begin(), Wr[i].end(), combined.begin(), br[i]));
        ht_hat[i] = tanh(std::inner_product(Wh[i].begin(), Wh[i].end(), combined.begin(), bh[i]));
    }

    // Estado oculto
    std::vector<double> ht(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        ht[i] = (1 - zt[i]) * prev_h[i] + zt[i] * ht_hat[i];
    }

    return ht; // Retornar el estado oculto
}
