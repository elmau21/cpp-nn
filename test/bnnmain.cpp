#include <iostream>
#include "../src/BNN/BayesianNN.h"

int main(){
    int input_size = 2;
    int hidden_size = 3;
    int output_size = 1;


    BayesianNN model(input_size, hidden_size, output_size);

    std::vector<std::vector<double>> X = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    std::vector<double> y = { 0.0, 1.0, 1.0, 0.0}; //XOR Problem

    model.train(X, y, 1000, 0.01);

    std::cout << "Prediction for (1, 1): " << model.predict({1.0, 1.0}) << std::endl;

    return 0;
}