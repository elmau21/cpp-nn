#include "HMM.h"
#include <iostream>
#include <vector>
#include <utility>

int main() {
    // Example POS-tagged corpus (word, POS-tag)
    std::vector<std::pair<std::string, std::string>> corpus = {
        {"The", "DET"}, {"cat", "NOUN"}, {"sat", "VERB"}, {"on", "PREP"}, {"the", "DET"}, {"mat", "NOUN"}
    };

    HMM hmm;
    hmm.train(corpus);
    
    // Test sentence
    std::vector<std::string> sentence = {"The", "dog", "sat", "on", "the", "couch"};
    std::vector<std::string> predictedTags = hmm.predict(sentence);
    
    // Output predicted tags
    for (size_t i = 0; i < sentence.size(); ++i) {
        std::cout << sentence[i] << " -> " << predictedTags[i] << "\n";
    }
    
    return 0;
}
