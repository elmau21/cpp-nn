#include "DatasetLoader.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

DatasetLoader::DatasetLoader(const std::string& images_path, const std::string& labels_path) 
    : currentIndex(0) {
    // Cargar imágenes en formato MNIST
    cv::Mat image_data = cv::imread(images_path, cv::IMREAD_GRAYSCALE);
    if (image_data.empty()) {
        std::cerr << "Error: No se pudo cargar las imágenes desde " << images_path << std::endl;
        exit(1);
    }

    // Cargar las etiquetas (MNIST)
    std::ifstream labels_file(labels_path, std::ios::binary);
    if (!labels_file.is_open()) {
        std::cerr << "Error: No se pudo cargar las etiquetas desde " << labels_path << std::endl;
        exit(1);
    }

    int magic_number = 0, num_labels = 0;
    labels_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = __builtin_bswap32(magic_number);  // Convertir de big-endian a little-endian
    labels_file.read((char*)&num_labels, sizeof(num_labels));
    num_labels = __builtin_bswap32(num_labels);

    labels.resize(num_labels);
    for (int i = 0; i < num_labels; ++i) {
        unsigned char temp = 0;
        labels_file.read((char*)&temp, sizeof(temp));
        labels[i] = (int)temp;
    }

    // Redimensionar las imágenes a un tamaño manejable y guardarlas en el vector
    images = image_data.reshape(0, num_labels);  // Convertir a matriz 1D de imágenes
}

std::vector<std::vector<double>> DatasetLoader::getNextImage() {
    cv::Mat image = images.row(currentIndex);
    std::vector<std::vector<double>> processed_image(28, std::vector<double>(28));
    
    // Convertir cada pixel de la imagen a escala de grises y normalizar a [0, 1]
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            processed_image[i][j] = image.at<unsigned char>(i * 28 + j) / 255.0;
        }
    }

    ++currentIndex;
    return processed_image;
}

int DatasetLoader::getNextLabel() {
    return labels[currentIndex++];
}

bool DatasetLoader::hasNext() {
    return currentIndex < labels.size();
}
