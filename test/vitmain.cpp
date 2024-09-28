#include "ViT.cpp"

// Testing the Vision Transformer
int vitmain() {
    // Hyperparameters
    int img_height = 32;     // Height of the input image
    int img_width = 32;      // Width of the input image
    int patch_size = 8;      // Size of each patch
    int d_model = 128;       // Dimension of model
    int num_heads = 4;       // Number of attention heads
    int num_layers = 6;      // Number of transformer layers
    int hidden_dim = 256;    // Hidden dimension for feed-forward network

    // Create a random image
    std::vector<std::vector<double>> image(img_height, std::vector<double>(img_width, 1.0));

    // Patch embedding
    PatchEmbedding patch_embedding(patch_size, d_model);
    auto patches = patch_embedding.embed(image);

    // Create Vision Transformer model
    VisionTransformer vit(num_layers, num_heads, d_model / num_heads, d_model, hidden_dim);

    // Forward pass
    std::vector<double> output = vit.forward(patches[0]);  // Process the first patch for simplicity

    // Output the result
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
