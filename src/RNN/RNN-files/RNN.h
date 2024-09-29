#ifndef RNN_H
#define RNN_H

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

// Helper function for scalar sigmoid
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Function for vector sigmoid
std::vector<double> sigmoid(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = sigmoid(x[i]); // Apply the scalar sigmoid function to each element
    }
    return result;
}

class RNN {
private:
    int input_size;
    int hidden_size;
    int output_size;
    
    std::vector<std::vector<double>> Wxh; // Weights for input to hidden
    std::vector<std::vector<double>> Whh; // Weights for hidden to hidden
    std::vector<std::vector<double>> Why; // Weights for hidden to output
    
    std::vector<double> bh; // Bias for hidden layer
    std::vector<double> by; // Bias for output layer

    // Initialize weights randomly
    void initWeights() {
        auto randomInit = [](int rows, int cols) {
            std::vector<std::vector<double>> weights(rows, std::vector<double>(cols));
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    weights[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0; // Random values between -1 and 1
                }
            }
            return weights;
        };

        Wxh = randomInit(hidden_size, input_size);
        Whh = randomInit(hidden_size, hidden_size);
        Why = randomInit(output_size, hidden_size);
        
        bh.resize(hidden_size);
        by.resize(output_size);
    }

public:
    RNN(int input_size, int hidden_size, int output_size);
    std::vector<double> forward(const std::vector<double>& input, std::vector<double>& hidden_state);
};

#endif // RNN_H
