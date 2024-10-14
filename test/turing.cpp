#include "../src/TuringMachine/TuringMachine.h"

int main() {
    TuringMachine tm;

    // Define transitions for recognizing balanced parentheses
    tm.addTransition("START", '(', "PUSH", ' ', 'R');
    tm.addTransition("START", ')', "POP", ' ', 'R');
    tm.addTransition("PUSH", '(', "PUSH", ' ', 'R');
    tm.addTransition("PUSH", ')', "POP", ' ', 'R');
    tm.addTransition("POP", '(', "POP", ' ', 'R');
    tm.addTransition("POP", ')', "POP", ' ', 'R');
    tm.addTransition("POP", ' ', "HALT", ' ', 'N'); // When we reach the end, check for balance

    // Input string of parentheses
    std::string input = "(()())";
    
    // Run the Turing machine
    bool result = tm.run(input);
    
    // Output result
    std::cout << "Input: " << input << "\n";
    std::cout << "Is balanced? " << (result ? "Yes" : "No") << "\n";

    return 0;
}
