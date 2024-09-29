#include "LSTM.h"
#include <numeric> // Agregar esta línea

// Constructor de LSTM
LSTM::LSTM(int input_size, int hidden_size)
    : input_size(input_size), hidden_size(hidden_size) {
    // Inicializar pesos y sesgos aleatorios
    Wf.resize(hidden_size, std::vector<double>(input_size + hidden_size));
    Wi.resize(hidden_size, std::vector<double>(input_size + hidden_size));
    Wc.resize(hidden_size, std::vector<double>(input_size + hidden_size));
    Wo.resize(hidden_size, std::vector<double>(input_size + hidden_size));

    bf.resize(hidden_size);
    bi.resize(hidden_size);
    bc.resize(hidden_size);
    bo.resize(hidden_size);

    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size + hidden_size; ++j) {
            Wf[i][j] = static_cast<double>(rand()) / RAND_MAX;
            Wi[i][j] = static_cast<double>(rand()) / RAND_MAX;
            Wc[i][j] = static_cast<double>(rand()) / RAND_MAX;
            Wo[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
        bf[i] = static_cast<double>(rand()) / RAND_MAX;
        bi[i] = static_cast<double>(rand()) / RAND_MAX;
        bc[i] = static_cast<double>(rand()) / RAND_MAX;
        bo[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

// Función de activación sigmoide
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Propagación hacia adelante
std::vector<double> LSTM::forward(const std::vector<double>& input, const std::vector<double>& prev_h, const std::vector<double>& prev_c) {
    std::vector<double> combined(input.size() + prev_h.size());
    std::copy(input.begin(), input.end(), combined.begin());
    std::copy(prev_h.begin(), prev_h.end(), combined.begin() + input.size());

    // Puertas
    std::vector<double> ft(hidden_size), it(hidden_size), ct(hidden_size), ot(hidden_size), ct_hat(hidden_size);
    
    for (int i = 0; i < hidden_size; ++i) {
        ft[i] = sigmoid(std::inner_product(Wf[i].begin(), Wf[i].end(), combined.begin(), bf[i]));
        it[i] = sigmoid(std::inner_product(Wi[i].begin(), Wi[i].end(), combined.begin(), bi[i]));
        ct_hat[i] = tanh(std::inner_product(Wc[i].begin(), Wc[i].end(), combined.begin(), bc[i]));
        ot[i] = sigmoid(std::inner_product(Wo[i].begin(), Wo[i].end(), combined.begin(), bo[i]));
    }

    // Estado de celda y estado oculto
    std::vector<double> ct(hidden_size), ht(hidden_size);
    for (int i = 0; i < hidden_size; ++i) {
        ct[i] = ft[i] * prev_c[i] + it[i] * ct_hat[i];
        ht[i] = ot[i] * tanh(ct[i]);
    }

    return ht; // Retornar el estado oculto
}
