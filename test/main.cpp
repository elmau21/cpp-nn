#include "NeuralNetwork.h"

int main() {
    // Define network structure: 2 inputs, 4 hidden neurons, 1 output
    NeuralNetwork nn(2, 4, 1);

    // XOR training data
    double inputData[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    };

    double outputData[4] = {0.0, 1.0, 1.0, 0.0};

    // Test the neural network with XOR inputs
    for (int i = 0; i < 4; ++i) {
        cout << "Input: (" << inputData[i][0] << ", " << inputData[i][1] << ")" << endl;
        nn.feedForward(inputData[i]);
        nn.printOutput();
        cout << "Expected Output: " << outputData[i] << endl;
        cout << "-----------------------------" << endl;
    }

    return 0;
}