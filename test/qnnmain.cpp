#include "QuantumNeuralNetwork.h"
#include <iostream>

int qnnmain() {
    QuantumNeuralNetwork qnn(4, 1);

    // Dummy data for testing
    std::vector<std::vector<double>> data = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}};
    std::vector<int> labels = {1, 0};

    qnn.train(data, labels, 100, 0.01);

    return 0;
}