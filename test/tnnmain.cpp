#include "Transformer.h"

int tnnmain() {
    int d_model = 512;         // Dimension of model
    int max_len = 60;         // Maximum length of input sequences
    int num_heads = 8;        // Number of attention heads
    int d_k = d_model / num_heads; // Dimension of keys
    int hidden_dim = 2048;    // Hidden dimension in feedforward layer
    int num_layers = 6;       // Number of encoder layers

    // Create the Transformer model
    Transformer transformer(num_layers, num_heads, d_k, d_model, hidden_dim);

    // Create a sample input (sequence of vectors)
    std::vector<double> sample_input(d_model, 1.0); // Example input with all ones

    // Forward pass through the Transformer
    std::vector<double> output = transformer.forward(sample_input);

    // Print the output
    std::cout << "Output from Transformer:" << std::endl;
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}