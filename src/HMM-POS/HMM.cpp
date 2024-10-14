#include "HMM.h"

// Constructor for the HMM class
HMM::HMM() {
    // Initialize with small probabilities (Laplacian smoothing can be applied during training)
}

// Train the HMM using a corpus of (word, tag) pairs
void HMM::train(const std::vector<std::pair<std::string, std::string>>& corpus) {
    // Maps to hold counts of tags and transitions/emissions
    std::unordered_map<std::string, int> tagCounts; // Count of each tag
    std::unordered_map<std::string, std::unordered_map<std::string, int>> transitionCounts; // Counts of transitions between tags
    std::unordered_map<std::string, std::unordered_map<std::string, int>> emissionCounts; // Counts of words emitted by tags

    std::string prevTag = ""; // Variable to store the previous tag

    // Iterate through each (word, tag) pair in the training corpus
    for (const auto& [word, tag] : corpus) {
        // Count initial tag probabilities
        if (prevTag == "") {
            initialProbs[tag]++; // Increment the count for the initial tag
        }

        // Count transition probabilities
        if (!prevTag.empty()) {
            transitionCounts[prevTag][tag]++; // Increment transition from prevTag to tag
        }

        // Count emission probabilities
        emissionCounts[tag][word]++; // Increment count of the word emitted by the tag
        tagCounts[tag]++; // Increment total count for the tag
        
        prevTag = tag; // Update the previous tag to the current one
    }

    // Calculate final probabilities using Maximum Likelihood Estimation (MLE)
    for (const auto& [tag1, tagTransitions] : transitionCounts) {
        for (const auto& [tag2, count] : tagTransitions) {
            // P(tag2 | tag1) = count(tag1, tag2) / count(tag1)
            transitionProbs[tag1][tag2] = static_cast<double>(count) / tagCounts[tag1]; // Calculate transition probability
        }
    }

    for (const auto& [tag, words] : emissionCounts) {
        for (const auto& [word, count] : words) {
            // P(word | tag) = count(tag, word) / count(tag)
            emissionProbs[tag][word] = static_cast<double>(count) / tagCounts[tag]; // Calculate emission probability
        }
    }

    // Normalize initial probabilities
    int totalSentences = initialProbs.size(); // Total number of unique initial tags
    for (auto& [tag, count] : initialProbs) {
        // P(tag) = count(tag) / total_sentences
        initialProbs[tag] = static_cast<double>(count) / totalSentences; // Normalize initial probabilities
    }
}

// Predict POS tags using the Viterbi algorithm
std::vector<std::string> HMM::predict(const std::vector<std::string>& sentence) {
    // Viterbi table to hold maximum probabilities
    std::vector<std::unordered_map<std::string, double>> viterbi(sentence.size());
    // Backpointer table to keep track of the best previous tag for backtracking
    std::vector<std::unordered_map<std::string, std::string>> backpointer(sentence.size());

    // Initialize for the first word
    for (const auto& [tag, prob] : initialProbs) {
        // log(P(tag)) + log(P(word | tag))
        viterbi[0][tag] = log(prob) + log(getEmissionProb(tag, sentence[0])); // Calculate the probability for the first word
        backpointer[0][tag] = ""; // No previous tag for the first word
    }

    // Dynamic programming for the rest of the words
    for (size_t t = 1; t < sentence.size(); ++t) {
        for (const auto& [currTag, _] : emissionProbs) {
            double maxProb = -INFINITY; // Start with the lowest possible probability
            std::string bestPrevTag; // Variable to store the best previous tag
            for (const auto& [prevTag, _] : viterbi[t-1]) {
                // Calculate the probability for the current tag given the previous tag
                double prob = viterbi[t-1][prevTag] + log(getTransitionProb(prevTag, currTag)) + log(getEmissionProb(currTag, sentence[t]));
                // Update if the current probability is higher than the maximum found so far
                if (prob > maxProb) {
                    maxProb = prob; // Update max probability
                    bestPrevTag = prevTag; // Update the best previous tag
                }
            }
            viterbi[t][currTag] = maxProb; // Store the max probability for the current tag
            backpointer[t][currTag] = bestPrevTag; // Store the best previous tag for backtracking
        }
    }

    // Backtrack to find the best sequence
    std::vector<std::string> bestSequence(sentence.size()); // Vector to hold the best sequence of tags
    double maxFinalProb = -INFINITY; // Initialize maximum final probability
    std::string bestFinalTag; // Variable to store the best final tag

    for (const auto& [tag, prob] : viterbi.back()) {
        // Find the tag with the highest probability for the last word
        if (prob > maxFinalProb) {
            maxFinalProb = prob; // Update maximum final probability
            bestFinalTag = tag; // Update the best final tag
        }
    }

    bestSequence.back() = bestFinalTag; // Set the last tag in the best sequence
    // Backtrack through the sequence
    for (int t = sentence.size() - 2; t >= 0; --t) {
        bestSequence[t] = backpointer[t + 1][bestSequence[t + 1]]; // Set the best tag for each position
    }

    return bestSequence; // Return the best sequence of tags
}

// Helper functions to retrieve probabilities
double HMM::getTransitionProb(const std::string& prevTag, const std::string& currTag) const {
    // Check if the transition probability exists
    if (transitionProbs.find(prevTag) != transitionProbs.end() && transitionProbs.at(prevTag).find(currTag) != transitionProbs.at(prevTag).end()) {
        return transitionProbs.at(prevTag).at(currTag); // Return the transition probability
    }
    return 1e-6; // Small probability for unseen transitions
}

double HMM::getEmissionProb(const std::string& tag, const std::string& word) const {
    // Check if the emission probability exists
    if (emissionProbs.find(tag) != emissionProbs.end() && emissionProbs.at(tag).find(word) != emissionProbs.at(tag).end()) {
        return emissionProbs.at(tag).at(word); // Return the emission probability
    }
    return 1e-6; // Small probability for unseen words
}

double HMM::getInitialProb(const std::string& tag) const {
    // Check if the initial probability exists
    if (initialProbs.find(tag) != initialProbs.end()) {
        return initialProbs.at(tag); // Return the initial probability
    }
    return 1e-6; // Small probability for unseen initial tags
}

void HMM::printModel() const {
    // Output transition probabilities
    std::cout << "Transition Probabilities:\n";
    for (const auto& [tag1, tagTransitions] : transitionProbs) {
        for (const auto& [tag2, prob] : tagTransitions) {
            std::cout << "P(" << tag2 << " | " << tag1 << ") = " << prob << "\n"; // Print the transition probability
        }
    }

    // Output emission probabilities
    std::cout << "Emission Probabilities:\n";
    for (const auto& [tag, words] : emissionProbs) {
        for (const auto& [word, prob] : words) {
            std::cout << "P(" << word << " | " << tag << ") = " << prob << "\n"; // Print the emission probability
        }
    }
}