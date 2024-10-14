#ifndef BOLTZMANNMACHINE_H
#define BOLTZMANNMACHINE_H

#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>

class BoltzmannMachine {
public:
    BoltzmannMachine(int numVisible, int numHidden);
    void train(const std::vector<std::vector<double>>& data, int epochs, double learningRate, int k);
    std::vector<double> sampleVisible(const std::vector<double>& hiddenStates);
    std::vector<double> sampleHidden(const std::vector<double>& visibleStates);
    void saveModel(const std::string& filename);
    void loadModel(const std::string& filename);
    
private:
    int numVisible;
    int numHidden;
    std::vector<std::vector<double>> weights; // Weight matrix
    std::vector<double> visibleBias;          // Visible biases
    std::vector<double> hiddenBias;           // Hidden biases

    void updateWeights(const std::vector<double>& v0, const std::vector<double>& v1, const std::vector<double>& h0, const std::vector<double>& h1, double learningRate);
    double sigmoid(double x);
};

#endif