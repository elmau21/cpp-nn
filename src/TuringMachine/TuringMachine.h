#ifndef TURING_MACHINE_H
#define TURING_MACHINE_H

#include <iostream>
#include <unordered_map>
#include <vector>
#include <string>
#include <tuple>

class TuringMachine {
public:
    // Constructor
    TuringMachine();
    
    // Method to add a transition
    void addTransition(const std::string& state, char readSymbol, const std::string& newState, char writeSymbol, char direction);
    
    // Method to run the machine with the given input
    bool run(const std::string& input);
    
    // Method to print the current tape
    void printTape() const;

private:
    // Map of transitions: current state and symbol -> new state, write symbol, direction (L or R)
    std::unordered_map<std::string, std::unordered_map<char, std::tuple<std::string, char, char>>> transitions;
    
    // The current state of the Turing machine
    std::string currentState;
    
    // The tape, a vector of characters
    std::string tape;
    
    // The current position of the head on the tape
    int headPosition;
    
    // A stack to track parentheses (for balancing parentheses problem)
    std::vector<char> stack;
};

#endif // TURING_MACHINE_H
