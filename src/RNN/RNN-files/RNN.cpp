#include "RNN.h"

// Constructor
RNN::RNN(int input_size, int hidden_size, int output_size)
    : input_size(input_size), hidden_size(hidden_size), output_size(output_size) {
    initWeights();
}

// Forward pass
std::vector<double> RNN::forward(const std::vector<double>& input, std::vector<double>& hidden_state) {
    // Calculate the new hidden state
    std::vector<double> new_hidden(hidden_size, 0);
    
    // Compute the new hidden state
    for (int i = 0; i < hidden_size; ++i) {
        for (int j = 0; j < input_size; ++j) {
            new_hidden[i] += Wxh[i][j] * input[j];
        }
        for (int j = 0; j < hidden_size; ++j) {
            new_hidden[i] += Whh[i][j] * hidden_state[j];
        }
        new_hidden[i] += bh[i];
        new_hidden[i] = sigmoid(new_hidden[i]); // Apply activation
    }

    // Calculate output
    std::vector<double> output(output_size, 0);
    for (int i = 0; i < output_size; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            output[i] += Why[i][j] * new_hidden[j];
        }
        output[i] += by[i];
        output[i] = sigmoid(output[i]); // Apply activation
    }

    // Update hidden state
    hidden_state = new_hidden;

    return output;
}