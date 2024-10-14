#include "TuringMachine.h"
#include <iostream>

// Constructor
TuringMachine::TuringMachine() : currentState("START"), headPosition(0) {
    tape.resize(100, ' '); // Initialize tape with blanks
}

// Add a transition to the Turing machine
void TuringMachine::addTransition(const std::string& state, char readSymbol, const std::string& newState, char writeSymbol, char direction) {
    transitions[state][readSymbol] = std::make_tuple(newState, writeSymbol, direction);
}

// Run the Turing machine with the provided input
bool TuringMachine::run(const std::string& input) {
    // Load input onto the tape
    for (size_t i = 0; i < input.length(); ++i) {
        tape[i] = input[i];
    }

    std::cout << "Initial tape: " << tape.substr(0, input.length()) << "\n";
    std::cout << "Starting state: " << currentState << "\n";

    while (currentState != "HALT") {
        char currentSymbol = tape[headPosition];

        // Print the current state and tape information
        std::cout << "\nStep:\n";
        std::cout << "Current state: " << currentState << "\n";
        std::cout << "Head position: " << headPosition << "\n";
        std::cout << "Read symbol: " << currentSymbol << "\n";
        std::cout << "Tape: " << tape.substr(0, input.length()) << "\n";
        
        // Check if a transition exists
        if (transitions.find(currentState) != transitions.end() && transitions[currentState].find(currentSymbol) != transitions[currentState].end()) {
            auto [newState, writeSymbol, direction] = transitions[currentState][currentSymbol];
            
            // Write to tape
            tape[headPosition] = writeSymbol;
            currentState = newState;

            // Print the write operation and direction
            std::cout << "Write symbol: " << writeSymbol << "\n";
            std::cout << "Move direction: " << (direction == 'R' ? "Right" : "Left") << "\n";
            
            // Handle stack operations for balancing parentheses
            if (currentSymbol == '(') {
                stack.push_back('('); // Push onto the stack
            } else if (currentSymbol == ')') {
                if (!stack.empty()) {
                    stack.pop_back(); // Pop from the stack if possible
                } else {
                    std::cout << "Unmatched closing parenthesis\n";
                    return false; // Unmatched closing parenthesis
                }
            }
            
            // Move the head
            if (direction == 'R') {
                headPosition++;
            } else if (direction == 'L') {
                if (headPosition > 0) headPosition--;
            }
        } else {
            // No valid transition, halt the machine
            std::cout << "No valid transition found. Halting.\n";
            currentState = "HALT";
        }
    }
    
    // Check if stack is empty (balanced parentheses)
    if (stack.empty()) {
        std::cout << "Input is balanced.\n";
    } else {
        std::cout << "Unmatched opening parentheses.\n";
    }

    return stack.empty();
}

// Print the current state of the tape
void TuringMachine::printTape() const {
    for (const auto& symbol : tape) {
        std::cout << symbol;
    }
    std::cout << "\n";
}