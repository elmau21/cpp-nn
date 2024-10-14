#include <iostream>
#include <vector>
#include "BoltzmannMachine.h"

int main() {
    // Initialize the Boltzmann Machine with the number of visible and hidden units
    int numVisible = 6;  // Example number of visible units
    int numHidden = 2;   // Example number of hidden units
    BoltzmannMachine bm(numVisible, numHidden);

    // Example binary training data (should be of type double)
    std::vector<std::vector<double>> trainingData = {
        {1.0, 0.0, 1.0, 0.0, 1.0, 0.0},
        {0.0, 1.0, 0.0, 1.0, 0.0, 1.0},
        {1.0, 1.0, 0.0, 0.0, 1.0, 1.0},
        {0.0, 0.0, 1.0, 1.0, 0.0, 0.0}
    };

    // Train the Boltzmann Machine
    int epochs = 1000;     // Number of training epochs
    double learningRate = 0.1; // Learning rate for weight updates
    int k = 1;            // Number of contrastive divergence steps

    bm.train(trainingData, epochs, learningRate, k);

    // Test the Boltzmann Machine by sampling from the trained model
    for (const auto& sample : trainingData) {
        std::vector<double> hiddenStates = bm.sampleHidden(sample);
        std::vector<double> reconstructedVisible = bm.sampleVisible(hiddenStates);

        std::cout << "Original: ";
        for (const auto& value : sample) {
            std::cout << value << " ";
        }
        std::cout << "-> Reconstructed: ";
        for (const auto& value : reconstructedVisible) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}