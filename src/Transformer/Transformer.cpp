#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric> // Include for std::accumulate

// Helper functions
double dotProduct(const std::vector<double>& a, const std::vector<double>& b) {
    double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

std::vector<double> softmax(const std::vector<double>& values) {
    std::vector<double> softmax_values(values.size());
    double max_val = *std::max_element(values.begin(), values.end());
    double sum_exp = 0;

    for (size_t i = 0; i < values.size(); ++i) {
        softmax_values[i] = exp(values[i] - max_val);
        sum_exp += softmax_values[i];
    }

    for (size_t i = 0; i < values.size(); ++i) {
        softmax_values[i] /= sum_exp;
    }

    return softmax_values;
}

// Positional Encoding
class PositionalEncoding {
private:
    int d_model;
    std::vector<std::vector<double>> pos_encodings;

public:
    PositionalEncoding(int d_model, int max_len) : d_model(d_model) {
        pos_encodings.resize(max_len, std::vector<double>(d_model));

        for (int pos = 0; pos < max_len; ++pos) {
            for (int i = 0; i < d_model; ++i) {
                if (i % 2 == 0) {
                    pos_encodings[pos][i] = sin(pos / pow(10000.0, 2.0 * i / d_model));
                } else {
                    pos_encodings[pos][i] = cos(pos / pow(10000.0, 2.0 * (i - 1) / d_model));
                }
            }
        }
    }

    const std::vector<double>& getEncoding(int pos) const {
        return pos_encodings[pos];
    }
};

// Self-Attention
class SelfAttention {
private:
    int d_k;

public:
    SelfAttention(int d_k) : d_k(d_k) {}

    std::vector<double> attention(const std::vector<double>& Q,
                                  const std::vector<double>& K,
                                  const std::vector<double>& V) {
        double score = dotProduct(Q, K) / sqrt(d_k);
        std::vector<double> weights = softmax({score});

        std::vector<double> output(V.size(), 0);
        for (size_t i = 0; i < V.size(); ++i) {
            output[i] = weights[0] * V[i];
        }

        return output;
    }
};

// Multi-Head Attention
class MultiHeadAttention {
private:
    int num_heads;
    int d_k;
    std::vector<SelfAttention> heads;

public:
    MultiHeadAttention(int num_heads, int d_k) : num_heads(num_heads), d_k(d_k) {
        for (int i = 0; i < num_heads; ++i) {
            heads.emplace_back(d_k);
        }
    }

    std::vector<double> apply(const std::vector<std::vector<double>>& Q,
                              const std::vector<std::vector<double>>& K,
                              const std::vector<std::vector<double>>& V) {
        std::vector<double> output(Q[0].size(), 0);
        for (int i = 0; i < num_heads; ++i) {
            std::vector<double> head_output = heads[i].attention(Q[i], K[i], V[i]);
            for (size_t j = 0; j < output.size(); ++j) {
                output[j] += head_output[j];  // Simple concatenation
            }
        }
        return output;
    }
};

// Feedforward Layer
class FeedForwardNetwork {
private:
    int input_dim;
    int hidden_dim;
    std::vector<double> W1;
    std::vector<double> W2;
    std::vector<double> b1;
    std::vector<double> b2;

public:
    FeedForwardNetwork(int input_dim, int hidden_dim)
        : input_dim(input_dim), hidden_dim(hidden_dim) {
        W1.resize(input_dim * hidden_dim);
        W2.resize(hidden_dim * input_dim);
        b1.resize(hidden_dim);
        b2.resize(input_dim);
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> hidden(hidden_dim, 0);
        for (int i = 0; i < hidden_dim; ++i) {
            hidden[i] = std::max(0.0, dotProduct(input, W1) + b1[i]);  // ReLU
        }

        std::vector<double> output(input_dim, 0);
        for (int i = 0; i < input_dim; ++i) {
            output[i] = dotProduct(hidden, W2) + b2[i];
        }

        return output;
    }
};

// Layer Normalization
class LayerNormalization {
public:
    std::vector<double> normalize(const std::vector<double>& input) {
        double mean = std::accumulate(input.begin(), input.end(), 0.0) / input.size();
        double variance = std::accumulate(input.begin(), input.end(), 0.0, [mean](double accum, double val) {
            return accum + (val - mean) * (val - mean);
        }) / input.size();

        std::vector<double> normalized(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            normalized[i] = (input[i] - mean) / sqrt(variance + 1e-6);
        }

        return normalized;
    }
};

// Transformer Encoder Layer
class TransformerEncoderLayer {
private:
    MultiHeadAttention attention;
    FeedForwardNetwork ffn;
    LayerNormalization norm1;
    LayerNormalization norm2;

public:
    TransformerEncoderLayer(int num_heads, int d_k, int input_dim, int hidden_dim)
        : attention(num_heads, d_k), ffn(input_dim, hidden_dim) {}

    std::vector<double> forward(const std::vector<double>& input) {
        // Self-Attention
        auto attn_output = attention.apply({input}, {input}, {input});

        // Add & Norm
        std::vector<double> norm1_output = norm1.normalize(attn_output);
        std::vector<double> ffn_output = ffn.forward(norm1_output);

        // Add & Norm
        return norm2.normalize(ffn_output);
    }
};

// Transformer Model
class Transformer {
private:
    std::vector<TransformerEncoderLayer> encoder_layers;

public:
    Transformer(int num_layers, int num_heads, int d_k, int input_dim, int hidden_dim) {
        for (int i = 0; i < num_layers; ++i) {
            encoder_layers.emplace_back(num_heads, d_k, input_dim, hidden_dim);
        }
    }

    std::vector<double> forward(const std::vector<double>& input) {
        std::vector<double> output = input;
        for (auto& layer : encoder_layers) {
            output = layer.forward(output);
        }
        return output;
    }
};