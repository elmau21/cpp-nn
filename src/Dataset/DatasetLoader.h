#ifndef DATASETLOADER_H
#define DATASETLOADER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

class DatasetLoader {
public:
    DatasetLoader(const std::string& images_path, const std::string& labels_path);
    std::vector<std::vector<double>> getNextImage();
    int getNextLabel();
    bool hasNext();

private:
    cv::Mat images;
    std::vector<int> labels;
    size_t currentIndex;
};

#endif
