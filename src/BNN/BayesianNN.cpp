#include "BayesianNN.h"
#include <iostream>
#include <numeric>

BayesianNN::BayesianNN(int input_size, int hidden_size, int output_size)
    : input_size_(input_size), hidden_size_(hidden_size), output_size_(output_size),
      weights_input_hidden_mean_(input_size * hidden_size), weights_input_hidden_logvar_(input_size * hidden_size),
      weights_hidden_output_mean_(hidden_size * output_size), weights_hidden_output_logvar_(hidden_size * output_size),
      distribution_(0.0, 1.0) {
    
    // Initialize weights with small mean and log variance
    for (double &weight : weights_input_hidden_mean_) {
        weight = distribution_(generator_) * 0.1; // Small initialization
    }
    for (double &logvar : weights_input_hidden_logvar_) {
        logvar = distribution_(generator_) * -5.0; // Small log variance (larger uncertainty)
    }
    for (double &weight : weights_hidden_output_mean_) {
        weight = distribution_(generator_) * 0.1;
    }
    for (double &logvar : weights_hidden_output_logvar_) {
        logvar = distribution_(generator_) * -5.0;
    }
}

double BayesianNN::activation(double x) {
    // Sigmoid activation function
    return 1.0 / (1.0 + exp(-x));
}

double BayesianNN::forward(const std::vector<double>& x) {
    // Forward pass with sampling
    std::vector<double> hidden(hidden_size_);
    std::vector<double> weights_input_hidden(weights_input_hidden_mean_.size());
    
    // Sample weights from the variational distribution
    for (size_t i = 0; i < weights_input_hidden.size(); ++i) {
        std::normal_distribution<double> sample(weights_input_hidden_mean_[i], exp(0.5 * weights_input_hidden_logvar_[i]));
        weights_input_hidden[i] = sample(generator_);
    }

    // Calculate hidden layer activations
    for (int i = 0; i < hidden_size_; ++i) {
        hidden[i] = 0;
        for (int j = 0; j < input_size_; ++j) {
            hidden[i] += x[j] * weights_input_hidden[j + i * input_size_];
        }
        hidden[i] = activation(hidden[i]);
    }

    // Similar process for the output layer
    double output = 0;
    std::vector<double> weights_hidden_output(weights_hidden_output_mean_.size());
    
    for (size_t i = 0; i < weights_hidden_output.size(); ++i) {
        std::normal_distribution<double> sample(weights_hidden_output_mean_[i], exp(0.5 * weights_hidden_output_logvar_[i]));
        weights_hidden_output[i] = sample(generator_);
    }

    for (int i = 0; i < output_size_; ++i) {
        output += hidden[i] * weights_hidden_output[i];
    }

    return activation(output);
}

double BayesianNN::kl_divergence(const std::vector<double>& mean, const std::vector<double>& logvar) {
    double kl = 0.0;
    for (size_t i = 0; i < mean.size(); ++i) {
        kl += 0.5 * (logvar[i] - log(mean[i] * mean[i]) - 1);
    }
    return kl;
}

void BayesianNN::update_weights(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double learning_rate) {
    // Placeholder for weight update logic using backpropagation and stochastic gradient descent
    for (size_t i = 0; i < X.size(); ++i) {
        double prediction = forward(X[i]);
        double loss = prediction - y[i];

        // Backpropagation steps to update weights would go here

        // Update means and log variances using gradients (simplified)
        for (size_t j = 0; j < weights_input_hidden_mean_.size(); ++j) {
            weights_input_hidden_mean_[j] -= learning_rate * (loss); // Placeholder for gradient
        }
        for (size_t j = 0; j < weights_hidden_output_mean_.size(); ++j) {
            weights_hidden_output_mean_[j] -= learning_rate * (loss); // Placeholder for gradient
        }

        // Calculate KL divergence and update log variances accordingly
        double kl = kl_divergence(weights_input_hidden_mean_, weights_input_hidden_logvar_);
        // Update log variances based on kl divergence (placeholder logic)
        for (size_t j = 0; j < weights_input_hidden_logvar_.size(); ++j) {
            weights_input_hidden_logvar_[j] -= learning_rate * (kl); // Placeholder for gradient
        }
    }
}

void BayesianNN::train(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int epochs, double learning_rate) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        update_weights(X, y, learning_rate);
    }
}

double BayesianNN::predict(const std::vector<double>& x) {
    return forward(x);
}