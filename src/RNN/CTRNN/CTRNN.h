#ifndef CTRNN_H
#define CTRNN_H

#include <vector>
#include <cmath>
#include <iostream>

class CTRNN {
public:
    int num_neurons;
    std::vector<double> u;        // State of each neuron
    std::vector<double> I;        // External input
    std::vector<std::vector<double>> W;  // Weight matrix
    double tau;                   // Time constant
    double dt;                    // Time step for integration

    // Constructor
    CTRNN(int n, double tau_val, double dt_val) : num_neurons(n), tau(tau_val), dt(dt_val) {
        u.resize(n, 0.0);
        I.resize(n, 0.0);
        W.resize(n, std::vector<double>(n, 0.0));
    }

    // Activation function (sigmoid)
    double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Update neuron states using Euler integration
    void update() {
        std::vector<double> u_new(num_neurons, 0.0);
        
        for (int i = 0; i < num_neurons; ++i) {
            double input_sum = 0.0;
            for (int j = 0; j < num_neurons; ++j) {
                input_sum += W[i][j] * sigmoid(u[j]);
            }
            u_new[i] = u[i] + dt * (-u[i] + input_sum + I[i]) / tau;
        }
        
        u = u_new;  // Update the state
    }

    // Set external input for neuron i
    void setInput(int i, double input) {
        I[i] = input;
    }

    // Print current state of neurons
    void printState() {
        for (int i = 0; i < num_neurons; ++i) {
            std::cout << "Neuron " << i << ": " << u[i] << std::endl;
        }
    }
};

#endif