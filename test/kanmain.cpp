#include <iostream>
#include "../src/KAN/KAN.h" // Asegúrate de que la ruta es correcta

int main() {
    // Tamaño de entrada, capa oculta y salida
    int input_size = 3;
    int hidden_size = 5;
    int output_size = 2;

    // Crear una KAN
    KAN kan(input_size, hidden_size, output_size);

    // Entrada de prueba
    std::vector<double> input = {1.0, 0.5, -1.5};

    // Realizar la propagación hacia adelante
    std::vector<double> output = kan.forward(input);

    // Mostrar la salida
    std::cout << "Salida de la KAN: ";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
