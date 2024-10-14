#include "BoltzmannMachine.h"

BoltzmannMachine::BoltzmannMachine(int numVisible, int numHidden)
    : numVisible(numVisible), numHidden(numHidden) {
    weights.resize(numVisible, std::vector<double>(numHidden));
    visibleBias.resize(numVisible);
    hiddenBias.resize(numHidden);

    // Initialize weights randomly
    for (int i = 0; i < numVisible; ++i) {
        for (int j = 0; j < numHidden; ++j) {
            weights[i][j] = static_cast<double>(rand()) / RAND_MAX * 0.1; // Small initialization
        }
    }

    // Initialize biases randomly
    for (int i = 0; i < numVisible; ++i) {
        visibleBias[i] = static_cast<double>(rand()) / RAND_MAX * 0.1;
    }
    for (int i = 0; i < numHidden; ++i) {
        hiddenBias[i] = static_cast<double>(rand()) / RAND_MAX * 0.1;
    }
}

double BoltzmannMachine::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

std::vector<double> BoltzmannMachine::sampleHidden(const std::vector<double>& visibleStates) {
    std::vector<double> hiddenProb(numHidden);
    for (int j = 0; j < numHidden; ++j) {
        double activation = hiddenBias[j];
        for (int i = 0; i < numVisible; ++i) {
            activation += visibleStates[i] * weights[i][j];
        }
        hiddenProb[j] = sigmoid(activation);
    }

    std::vector<double> hiddenStates(numHidden);
    for (size_t j = 0; j < hiddenProb.size(); ++j) {
        hiddenStates[j] = (rand() / static_cast<double>(RAND_MAX)) < hiddenProb[j] ? 1 : 0; // Sampling activation
    }
    return hiddenStates;
}

std::vector<double> BoltzmannMachine::sampleVisible(const std::vector<double>& hiddenStates) {
    std::vector<double> visibleProb(numVisible);
    for (int i = 0; i < numVisible; ++i) {
        double activation = visibleBias[i];
        for (int j = 0; j < numHidden; ++j) {
            activation += hiddenStates[j] * weights[i][j];
        }
        visibleProb[i] = sigmoid(activation);
    }

    std::vector<double> visibleStates(numVisible);
    for (size_t i = 0; i < visibleProb.size(); ++i) {
        visibleStates[i] = (rand() / static_cast<double>(RAND_MAX)) < visibleProb[i] ? 1 : 0; // Sampling activation
    }
    return visibleStates;
}

// Train the model
void BoltzmannMachine::train(const std::vector<std::vector<double>>& data, int epochs, double learningRate, int k) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::cout << "Epoch " << epoch + 1 << " / " << epochs << ":\n";
        for (const auto& sample : data) {
            std::vector<double> hiddenStates = sampleHidden(sample);
            std::vector<double> v1 = sampleVisible(hiddenStates);

            std::cout << "  Original: ";
            for (const auto& value : sample) std::cout << value << " ";
            std::cout << "-> Hidden states: ";
            for (const auto& value : hiddenStates) std::cout << value << " ";
            std::cout << "-> Reconstructed: ";
            for (const auto& value : v1) std::cout << value << " ";
            std::cout << "\n";

            for (int step = 0; step < k; ++step) {
                hiddenStates = sampleHidden(v1);
                v1 = sampleVisible(hiddenStates);
            }
            updateWeights(sample, v1, hiddenStates, hiddenStates, learningRate);
        }
        std::cout << "\n"; // Add spacing between epochs
    }
}

// Update weights and biases
void BoltzmannMachine::updateWeights(const std::vector<double>& v0, const std::vector<double>& v1, const std::vector<double>& h0, const std::vector<double>& h1, double learningRate) {
    for (int i = 0; i < numVisible; ++i) {
        for (int j = 0; j < numHidden; ++j) {
            weights[i][j] += learningRate * (v0[i] * h0[j] - v1[i] * h1[j]);
        }
        visibleBias[i] += learningRate * (v0[i] - v1[i]);
    }
    for (int j = 0; j < numHidden; ++j) {
        hiddenBias[j] += learningRate * (h0[j] - h1[j]);
    }
}

void BoltzmannMachine::saveModel(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file to save the model." << std::endl;
        return;
    }

    // Save weights
    for (const auto& row : weights) {
        for (double weight : row) {
            file.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        }
    }
    
    // Save biases
    for (double bias : visibleBias) {
        file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
    }
    for (double bias : hiddenBias) {
        file.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
    }

    file.close();
}

void BoltzmannMachine::loadModel(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file to load the model." << std::endl;
        return;
    }

    // Load weights
    for (auto& row : weights) {
        for (double& weight : row) {
            file.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        }
    }
    
    // Load biases
    for (double& bias : visibleBias) {
        file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
    }
    for (double& bias : hiddenBias) {
        file.read(reinterpret_cast<char*>(&bias), sizeof(bias));
    }

    file.close();
}