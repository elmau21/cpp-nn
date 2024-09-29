#include "CNN.h"

// Capa de convolución
ConvLayer::ConvLayer(int kernel_size, int stride) 
    : kernel_size(kernel_size), stride(stride) {
    // Inicializar kernel con valores aleatorios
    kernel.resize(kernel_size, std::vector<double>(kernel_size));
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            kernel[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
    }
}

std::vector<std::vector<double>> ConvLayer::forward(const std::vector<std::vector<double>>& input) {
    int output_size = (input.size() - kernel_size) / stride + 1;
    std::vector<std::vector<double>> output(output_size, std::vector<double>(output_size));

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            double sum = 0.0;
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    sum += input[i * stride + ki][j * stride + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = std::tanh(sum);  // Función de activación tanh
        }
    }
    return output;
}

// Capa de pooling
PoolingLayer::PoolingLayer(int pool_size) : pool_size(pool_size) {}

std::vector<std::vector<double>> PoolingLayer::forward(const std::vector<std::vector<double>>& input) {
    int output_size = input.size() / pool_size;
    std::vector<std::vector<double>> output(output_size, std::vector<double>(output_size));

    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            double max_val = -INFINITY;
            for (int pi = 0; pi < pool_size; ++pi) {
                for (int pj = 0; pj < pool_size; ++pj) {
                    max_val = std::max(max_val, input[i * pool_size + pi][j * pool_size + pj]);
                }
            }
            output[i][j] = max_val;
        }
    }
    return output;
}

// Capa totalmente conectada
FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) {
    weights.resize(output_size, std::vector<double>(input_size));
    biases.resize(output_size);

    // Inicializar pesos y sesgos aleatorios
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            weights[i][j] = static_cast<double>(rand()) / RAND_MAX;
        }
        biases[i] = static_cast<double>(rand()) / RAND_MAX;
    }
}

std::vector<double> FullyConnectedLayer::forward(const std::vector<double>& input) {
    std::vector<double> output(biases.size());

    for (int i = 0; i < output.size(); ++i) {
        double sum = biases[i];
        for (int j = 0; j < input.size(); ++j) {
            sum += input[j] * weights[i][j];
        }
        output[i] = std::tanh(sum);  // Función de activación tanh
    }

    return output;
}

// CNN completa
CNN::CNN() : conv(3, 1), pool(2), fc(16, 10) {}

std::vector<double> CNN::forward(const std::vector<std::vector<double>>& input) {
    // Propagación hacia adelante a través de las capas
    std::vector<std::vector<double>> conv_output = conv.forward(input);
    std::vector<std::vector<double>> pool_output = pool.forward(conv_output);

    // Aplanar la salida de la capa de pooling
    std::vector<double> flattened;
    for (const auto& row : pool_output) {
        for (const auto& val : row) {
            flattened.push_back(val);
        }
    }

    return fc.forward(flattened);
}
