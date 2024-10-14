#ifndef BAYESIANNN_H
#define BAYESIANNN_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>

class BayesianNN {
    public: 
    BayesianNN(int input_size, int hidden_size, int output_size);
    void train(const std::vector<std::vector<double>>& X, const std::vector<double>& y, int epochs, double learning_rate);
    double predict(const std::vector<double>& x);

    private:
    int input_size_;
    int hidden_size_;
    int output_size_;

    std::vector<double> weights_input_hidden_mean_;
    std::vector<double> weights_input_hidden_logvar_;
    std::vector<double> weights_hidden_output_mean_;
    std::vector<double> weights_hidden_output_logvar_;


    std::default_random_engine generator_;
    std::normal_distribution<double> distribution_;


    double activation(double x);
    double forward(const std::vector<double>& x);
    double kl_divergence(const std::vector<double>& mean, const std::vector<double>& logvar);
    void update_weights(const std::vector<std::vector<double>>& X, const std::vector<double>& y, double learning_rate);
};

#endif // BAYESIANNN_H