#include "LiquidNeuralNetwork.h"
#include <iostream>
#include <vector>

int lnnmain() {
    // Training data (you can modify this data with something more realistic)
    std::vector<std::vector<double>> training_data = {
        {0.1, 0.2, 0.3},
        {0.4, 0.5, 0.6},
        {0.7, 0.8, 0.9}
    };

    // Labels for the training data
    std::vector<double> labels = {0.2, 0.5, 0.8};

    // Create a liquid neural network with 3 neurons and a time constant of 1.0
    int num_neurons = 3;
    double time_constant = 1.0;
    LiquidNeuralNetwork lnn(num_neurons, time_constant);

    // Training parameters
    double delta_t = 0.1;   // Time step
    int epochs = 100;       // Number of training epochs
    double learning_rate = 0.01;  // Learning rate

    // Train the liquid neural network
    std::cout << "Training the liquid neural network..." << std::endl;
    lnn.train(training_data, labels, delta_t, epochs, learning_rate);

    // Make predictions
    std::cout << "Prediction after training:" << std::endl;
    for (const auto& sample : training_data) {
        std::vector<double> outputs = lnn.forward(sample, delta_t);
        double final_output = lnn.computeFinalOutput(outputs);
        std::cout << "Input: [";
        for (const auto& val : sample) {
            std::cout << val << " ";
        }
        std::cout << "] -> Prediction: " << final_output << std::endl;
    }

    return 0;
}
