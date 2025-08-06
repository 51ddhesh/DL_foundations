#include <iostream>
#include <array>
#include <vector>
#include <random>
#include <cmath>
#include <format>

constexpr int input_size = 2;
constexpr int hidden_size = 2;
constexpr int output_size = 1;
constexpr double learning_rate = 0.1;
constexpr int epochs = 10'000;

constexpr std::array<std::array<double, input_size>, 4> x = {{
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
}};

constexpr std::array<double, 4> y = {0, 1, 1, 0}; // target

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

double random_weight() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

int main() {
    std::array<std::array<double, hidden_size>, input_size> weights_input_hidden {};
    std::array<double, hidden_size> hidden_output {};
    std::array<double, hidden_size> bias_hidden {};
    
    std::array<double, hidden_size> d_hidden {};
    std::array<double, hidden_size> hidden_input {};


    std::array<double, hidden_size> weights_hidden_output{};
    double bias_output = random_weight();
    double final_output = 0.0;
    
    
    for (size_t i = 0; i < input_size; ++i)
        for (size_t j = 0; j < hidden_size; ++j)
            weights_input_hidden[i][j] = random_weight();


    for (auto& b : bias_hidden) b = random_weight();
    for (auto& w : weights_hidden_output) w = random_weight();

    
    // Training Loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        // Forward Pass
        for (size_t i = 0; i < 4; i++) {
            for (size_t j = 0; j < hidden_size; j++) {
                hidden_input[j] = x[i][0] * weights_input_hidden[0][j] + \
                                  x[i][1] * weights_input_hidden[1][j] + \
                                  bias_hidden[j];
                hidden_output[j] = sigmoid(hidden_input[j]);
            }

            double final_input = hidden_output[0] * weights_hidden_output[0] + hidden_output[1] * weights_hidden_output[1] + bias_output;

            final_output = sigmoid(final_input);
        
            double error = y[i] - final_output;
            total_loss += error * error;
        
            // Backward pass
            double d_output = error * sigmoid_derivative(final_output);
            
            for (int j = 0; j < hidden_size; j++) {
                d_hidden[j] = d_output * weights_hidden_output[j] * sigmoid_derivative(hidden_output[j]);
            }

            // Update weights and biases to minimize loss
            for (size_t j = 0; j < hidden_size; j++) {
                weights_hidden_output[j] += learning_rate * d_output * hidden_output[j];
                for (size_t k = 0; k < input_size; k++) {
                    weights_input_hidden[k][j] += learning_rate * d_hidden[j] * x[i][k];
                }
                bias_hidden[j] += learning_rate * d_hidden[j];
            }
            bias_output += learning_rate * d_output;
        }
        if (epoch % 1000 == 0) std::printf("Epoch %5d, Loss: %.5f\n", epoch, total_loss / 4.0);
    }

    // Tests
    std::cout << "\n--- XOR Predictions ---\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            hidden_input[j] = x[i][0] * weights_input_hidden[0][j] + 
                              x[i][1] * weights_input_hidden[1][j] + 
                              bias_hidden[j];
            hidden_output[j] = sigmoid(hidden_input[j]);
        }

        double final_input = hidden_output[0] * weights_hidden_output[0] + 
                            hidden_output[1] * weights_hidden_output[1] + 
                            bias_output;
        final_output = sigmoid(final_input);

        std::printf("%d XOR %d = %.0f\n", (int)x[i][0], (int)x[i][1], round(final_output));
    }

}
