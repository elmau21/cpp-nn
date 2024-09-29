#include "SiameseNetwork.h"

int siamesemain() {
    const int input_dim = 10; // Example input dimension (length of DNA sequence)
    const int hidden_dim = 5;  // Hidden layer size
    const double learning_rate = 0.01;

    SiameseNetwork network(input_dim, hidden_dim, learning_rate);

    // Example training data (two pairs of sequences with corresponding labels)
    std::vector<std::vector<double>> inputs = {
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}, // Sequence 1
        {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9}  // Sequence 2
    };
    std::vector<double> targets = {1.0, 0.0}; // Labels for each sequence pair

    // Training the network
    network.train(inputs, targets);

    // Example prediction
    std::vector<double> new_input = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    double prediction = network.predict(new_input);
    std::cout << "Prediction: " << prediction << std::endl;

    return 0;
}
