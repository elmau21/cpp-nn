#ifndef HMM_H
#define HMM_H

#include <unordered_map>
#include <vector>
#include <string>
#include <iostream>
#include <cmath>

class HMM {
public:
    HMM();
    
    // Train the HMM using a tagged corpus
    void train(const std::vector<std::pair<std::string, std::string>>& corpus);
    
    // Predict POS tags for a sentence using the Viterbi algorithm
    std::vector<std::string> predict(const std::vector<std::string>& sentence);
    
    // Print the transition, emission, and initial probabilities
    void printModel() const;

private:
    std::unordered_map<std::string, std::unordered_map<std::string, double>> transitionProbs; // P(tag2 | tag1)
    std::unordered_map<std::string, std::unordered_map<std::string, double>> emissionProbs;   // P(word | tag)
    std::unordered_map<std::string, double> initialProbs; // P(tag)

    // Helper functions
    double getTransitionProb(const std::string& prevTag, const std::string& currTag) const;
    double getEmissionProb(const std::string& tag, const std::string& word) const;
    double getInitialProb(const std::string& tag) const;
};

#endif // HMM_H