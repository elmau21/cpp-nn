#include <iostream>
#include <vector>
#include "../src/CNN/CNN.h"
#include "../src/Dataset/DatasetLoader.h"

int main() {
    // Cargar dataset MNIST
    DatasetLoader loader("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte");

    // Crear la CNN
    CNN cnn;

    // Entrenamiento básico (solo se mostrará el resultado de la pasada hacia adelante)
    int num_samples = 100;
    for (int i = 0; i < num_samples && loader.hasNext(); ++i) {
        std::vector<std::vector<double>> input = loader.getNextImage();
        int label = loader.getNextLabel();

        // Realizar la propagación hacia adelante
        std::vector<double> output = cnn.forward(input);

        // Mostrar la salida y la etiqueta correspondiente
        std::cout << "Sample " << i + 1 << " - Label: " << label << " - Output: ";
        for (const auto& val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
