#ifndef SIAMESE_NETWORK_H
#define SIAMESE_NETWORK_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <numeric>
#include <algorithm>
#include <stdexcept>

class SiameseNetwork {
private:
    int input_dim;         // Dimension of input sequences
    int hidden_dim;        // Number of neurons in hidden layer
    double learning_rate;  // Learning rate for optimization

    std::vector<double> weights_hidden; // Weights for hidden layer
    std::vector<double> weights_output; // Weights for output layer

    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    void initializeWeights() {
        weights_hidden.resize(input_dim * hidden_dim);
        weights_output.resize(hidden_dim);
        
        for (auto &w : weights_hidden) {
            w = distribution(generator) * 0.01; // Small random values
        }
        for (auto &w : weights_output) {
            w = distribution(generator) * 0.01; // Small random values
        }
    }

    std::vector<double> sigmoid(const std::vector<double>& z) {
        std::vector<double> activated(z.size());
        std::transform(z.begin(), z.end(), activated.begin(), [](double val) {
            return 1.0 / (1.0 + std::exp(-val));
        });
        return activated;
    }

    std::vector<double> forward(const std::vector<double>& input) {
        if (input.size() != input_dim) {
            throw std::invalid_argument("Input size does not match the expected dimension.");
        }
        
        // Hidden layer
        std::vector<double> hidden(hidden_dim);
        for (int i = 0; i < hidden_dim; ++i) {
            hidden[i] = 0.0;
            for (int j = 0; j < input_dim; ++j) {
                hidden[i] += input[j] * weights_hidden[i * input_dim + j];
            }
        }
        hidden = sigmoid(hidden);

        // Output layer (single neuron)
        double output = 0.0;
        for (int i = 0; i < hidden_dim; ++i) {
            output += hidden[i] * weights_output[i];
        }

        return {output}; // Return output as vector for consistency
    }

public:
    SiameseNetwork(int input_dim, int hidden_dim, double learning_rate)
        : input_dim(input_dim), hidden_dim(hidden_dim), learning_rate(learning_rate),
          distribution(0.0, 1.0) {
        initializeWeights();
    }

    double computeLoss(const std::vector<double>& output, double target) {
        return 0.5 * std::pow(output[0] - target, 2); // Mean Squared Error
    }

    void train(const std::vector<std::vector<double>>& inputs,
               const std::vector<double>& targets) {
        if (inputs.size() != targets.size()) {
            throw std::invalid_argument("Input and target sizes do not match.");
        }

        for (size_t i = 0; i < inputs.size(); ++i) {
            auto output = forward(inputs[i]);
            double loss = computeLoss(output, targets[i]);

            // Backpropagation (simplified)
            double output_error = output[0] - targets[i];

            // Update weights for the output layer
            for (int j = 0; j < hidden_dim; ++j) {
                weights_output[j] -= learning_rate * output_error * output[j];
            }

            // Update weights for the hidden layer
            for (int j = 0; j < hidden_dim; ++j) {
                for (int k = 0; k < input_dim; ++k) {
                    weights_hidden[j * input_dim + k] -= learning_rate * output_error * inputs[i][k];
                }
            }

            std::cout << "Loss: " << loss << std::endl;
        }
    }

    double predict(const std::vector<double>& input) {
        auto output = forward(input);
        return output[0];
    }
};

#endif // SIAMESE_NETWORK_H
