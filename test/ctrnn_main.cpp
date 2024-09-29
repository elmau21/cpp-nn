#include <iostream>
#include "CTRNN.h"

int main() {
    // Crear una CTRNN con 3 neuronas, constante de tiempo tau=1.0, paso de tiempo dt=0.01
    CTRNN ctrnn(3, 1.0, 0.01);

    // Definir los pesos de la red
    ctrnn.W[0][1] = 0.5;  // Peso de la neurona 1 a la 0
    ctrnn.W[1][2] = 0.8;  // Peso de la neurona 2 a la 1
    ctrnn.W[2][0] = -0.4; // Peso de la neurona 0 a la 2

    // Establecer entradas externas
    ctrnn.setInput(0, 1.0);
    ctrnn.setInput(1, 0.5);
    ctrnn.setInput(2, 0.2);

    // Simular la CTRNN por 1000 pasos
    for (int t = 0; t < 1000; ++t) {
        ctrnn.update();
        if (t % 100 == 0) {  // Imprimir el estado cada 100 pasos
            std::cout << "Paso " << t << ":" << std::endl;
            ctrnn.printState();
        }
    }

    return 0;
}