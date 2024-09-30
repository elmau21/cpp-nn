#ifndef LIQUIDNEURALNETWORK_H
#define LIQUIDNEURALNETWORK_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

// Liquid Time-Constant Neuron
class LTCNeuron {
private:
    double time_constant;
    double state;
    double output;
    double weight_input;
    double weight_recurrent;

    double activationFunction(double x) {
        return tanh(x); // Non-linear activation
    }

public:
    LTCNeuron(double time_constant)
        : time_constant(time_constant), state(0.0), output(0.0) {
        initializeWeights();
    }

    void initializeWeights() {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 0.1);
        weight_input = distribution(generator);
        weight_recurrent = distribution(generator);
    }

    double forward(double input, double recurrent_input, double delta_t) {
        double input_contribution = weight_input * input;
        double recurrent_contribution = weight_recurrent * recurrent_input;

        // Dynamic state update based on time constant
        state += (-state + input_contribution + recurrent_contribution) * delta_t / time_constant;

        // Output through activation function
        output = activationFunction(state);

        return output;
    }

    double getState() const {
        return state;
    }

    double getOutput() const {
        return output;
    }
};

// Liquid Neural Network Class
class LiquidNeuralNetwork {
private:
    int num_neurons;
    std::vector<LTCNeuron> neurons;
    std::vector<double> output_layer_weights;

    void initializeOutputWeights() {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0, 0.1);
        output_layer_weights.resize(num_neurons);
        for (int i = 0; i < num_neurons; ++i) {
            output_layer_weights[i] = distribution(generator);
        }
    }

public:
    LiquidNeuralNetwork(int num_neurons, double time_constant)
        : num_neurons(num_neurons) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.emplace_back(time_constant);
        }
        initializeOutputWeights();
    }

    std::vector<double> forward(const std::vector<double>& input, double delta_t) {
        std::vector<double> outputs(num_neurons);
        for (int i = 0; i < num_neurons; ++i) {
            double recurrent_input = (i > 0) ? neurons[i - 1].getOutput() : 0.0;
            outputs[i] = neurons[i].forward(input[i], recurrent_input, delta_t);
        }
        return outputs;
    }

    double computeFinalOutput(const std::vector<double>& neuron_outputs) {
        double final_output = 0.0;
        for (int i = 0; i < num_neurons; ++i) {
            final_output += neuron_outputs[i] * output_layer_weights[i];
        }
        return final_output;
    }

    void train(const std::vector<std::vector<double>>& data, const std::vector<double>& labels, double delta_t, int epochs, double learning_rate) {
        std::cout << "Training Liquid Neural Network..." << std::endl;

        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            for (size_t i = 0; i < data.size(); ++i) {
                // Forward pass
                std::vector<double> neuron_outputs = forward(data[i], delta_t);
                double final_output = computeFinalOutput(neuron_outputs);

                // Compute loss (mean squared error)
                double loss = 0.5 * std::pow((final_output - labels[i]), 2);
                total_loss += loss;

                // Backpropagation (gradient descent on output layer)
                double error = final_output - labels[i];
                for (int j = 0; j < num_neurons; ++j) {
                    output_layer_weights[j] -= learning_rate * error * neuron_outputs[j];
                }
            }
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << " - Loss: " << total_loss / data.size() << std::endl;
        }
    }
};

#endif // LIQUIDNEURALNETWORK_H
