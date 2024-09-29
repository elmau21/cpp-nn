#ifndef CNN_H
#define CNN_H

#include <vector>
#include <iostream>
#include <cmath>
#include <cstdlib>

// Capa de convolución
class ConvLayer {
public:
    ConvLayer(int kernel_size, int stride);
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input);

private:
    int kernel_size;
    int stride;
    std::vector<std::vector<double>> kernel;
};

// Capa de pooling
class PoolingLayer {
public:
    PoolingLayer(int pool_size);
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input);

private:
    int pool_size;
};

// Capa totalmente conectada (fully connected)
class FullyConnectedLayer {
public:
    FullyConnectedLayer(int input_size, int output_size);
    std::vector<double> forward(const std::vector<double>& input);

private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

// CNN completa
class CNN {
public:
    CNN();
    std::vector<double> forward(const std::vector<std::vector<double>>& input);

private:
    ConvLayer conv;
    PoolingLayer pool;
    FullyConnectedLayer fc;
};

#endif