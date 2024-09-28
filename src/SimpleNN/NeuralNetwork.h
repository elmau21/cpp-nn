// NeuralNetwork.h

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Utility function to generate random numbers between -1 and 1
double randomWeight() {
    return ((double)rand() / RAND_MAX) * 2.0 - 1.0;
}

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Dot product of matrix and vector
void dot(double** matrix, double* vector, double* result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        result[i] = 0;
        for (int j = 0; j < cols; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// Add bias vector to result vector
void addBias(double* vector, double* bias, int size) {
    for (int i = 0; i < size; ++i) {
        vector[i] += bias[i];
    }
}

// Apply the activation function to each element of a vector
void applyActivation(double* vector, int size) {
    for (int i = 0; i < size; ++i) {
        vector[i] = sigmoid(vector[i]);
    }
}

// Neural Network class
class NeuralNetwork {
private:
    int inputNodes, hiddenNodes, outputNodes;
    double* inputLayer;
    double* hiddenLayer;
    double* outputLayer;
    double** weightsInputHidden;
    double** weightsHiddenOutput;
    double* hiddenBias;
    double* outputBias;

public:
    // Constructor
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes) {
        this->inputNodes = inputNodes;
        this->hiddenNodes = hiddenNodes;
        this->outputNodes = outputNodes;

        // Allocate memory for layers
        inputLayer = new double[inputNodes];
        hiddenLayer = new double[hiddenNodes];
        outputLayer = new double[outputNodes];

        // Allocate memory for weights and biases
        weightsInputHidden = new double*[hiddenNodes];
        for (int i = 0; i < hiddenNodes; ++i)
            weightsInputHidden[i] = new double[inputNodes];

        weightsHiddenOutput = new double*[outputNodes];
        for (int i = 0; i < outputNodes; ++i)
            weightsHiddenOutput[i] = new double[hiddenNodes];

        hiddenBias = new double[hiddenNodes];
        outputBias = new double[outputNodes];

        // Initialize weights and biases randomly
        initializeWeights();
    }

    // Initialize weights and biases randomly
    void initializeWeights() {
        srand((unsigned int)time(0));

        for (int i = 0; i < hiddenNodes; ++i)
            for (int j = 0; j < inputNodes; ++j)
                weightsInputHidden[i][j] = randomWeight();

        for (int i = 0; i < outputNodes; ++i)
            for (int j = 0; j < hiddenNodes; ++j)
                weightsHiddenOutput[i][j] = randomWeight();

        for (int i = 0; i < hiddenNodes; ++i)
            hiddenBias[i] = randomWeight();

        for (int i = 0; i < outputNodes; ++i)
            outputBias[i] = randomWeight();
    }

    // Feedforward function
    void feedForward(double* input) {
        // Load input
        for (int i = 0; i < inputNodes; ++i)
            inputLayer[i] = input[i];

        // Hidden layer computation: z_hidden = W_hidden * input + b_hidden
        dot(weightsInputHidden, inputLayer, hiddenLayer, hiddenNodes, inputNodes);
        addBias(hiddenLayer, hiddenBias, hiddenNodes);
        applyActivation(hiddenLayer, hiddenNodes);

        // Output layer computation: z_output = W_output * hidden + b_output
        dot(weightsHiddenOutput, hiddenLayer, outputLayer, outputNodes, hiddenNodes);
        addBias(outputLayer, outputBias, outputNodes);
        applyActivation(outputLayer, outputNodes);
    }

    // Print the output layer values
    void printOutput() {
        cout << "Output: ";
        for (int i = 0; i < outputNodes; ++i) {
            cout << outputLayer[i] << " ";
        }
        cout << endl;
    }

    // Destructor to free dynamically allocated memory
    ~NeuralNetwork() {
        delete[] inputLayer;
        delete[] hiddenLayer;
        delete[] outputLayer;

        for (int i = 0; i < hiddenNodes; ++i)
            delete[] weightsInputHidden[i];
        delete[] weightsInputHidden;

        for (int i = 0; i < outputNodes; ++i)
            delete[] weightsHiddenOutput[i];
        delete[] weightsHiddenOutput;

        delete[] hiddenBias;
        delete[] outputBias;
    }
};
