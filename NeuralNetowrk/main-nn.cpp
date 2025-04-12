#include <iostream>
#include <fstream>
#include "network/Network.hpp"

struct LearnData {
    uint32_t ExpectDigit;
    std::vector<double> Input;
};

std::vector<Eigen::VectorXd> load_images(const std::string &path, size_t count) {
    std::ifstream file(path);
    std::vector<Eigen::VectorXd> images;
    for (size_t i = 0; i < count; ++i) {
        Eigen::VectorXd img(784);
        for (int j = 0; j < 784; ++j) {
            double pixel;
            file >> pixel;
            pixel = static_cast<bool>(pixel);
            img(j) = pixel;
        }
        images.push_back(img);
    }
    return images;
}

std::vector<uint32_t> load_expects(const std::string &path, size_t count) {
    std::ifstream file(path);
    std::vector<uint32_t> labels(count);

    for (size_t i = 0; i < count; ++i) {
        file >> labels[i];
    }
    return labels;
}

int main() {
    NetworkProperty property;
    property.LayersCount = 3;
    property.NeuronsCount = {784, 256, 10};

    Network::Init(property);
    // Network::Init(property);
    std::vector<Eigen::VectorXd> train_images =
            load_images("C:/Drafts/github.com/neural_network/learn_data/train_images.txt", 60000);
    std::vector<uint32_t> train_labels = load_expects("C:/Drafts/github.com/neural_network/learn_data/train_expects.txt", 60000);
    std::vector<Eigen::VectorXd> test_images = load_images("C:/Drafts/github.com/neural_network/learn_data/test_images.txt", 10000);
    std::vector<uint32_t> test_labels = load_expects("C:/Drafts/github.com/neural_network/learn_data/test_expects.txt", 10000);

    std::cout << "Finished load files" << std::endl;

    double learningRate = 0.001f;
    uint32_t epochs = 30;

    for (uint32_t epoch = 0; epoch < epochs; ++epoch) {
        double error = 0.f;
        uint32_t correct = 0;

        for (size_t i = 0; i < train_images.size(); ++i) {
            Network::NeuronsValues[0] = train_images[i];
            uint32_t pred = Network::ForwardFeed();
            Eigen::VectorXd expected = Eigen::VectorXd::Zero(10);
            expected[train_labels[i]] = 1.0;

            Network::BackwardFeed(expected);
            // if (i % 1000 == 0) {
            //     std::cout << "Pred " << pred << " " << "Expect " << train_labels[i] << std::endl;
            // }

            for (uint32_t layer = 0; layer < Network::LayersCount - 1; ++layer) {
                Network::LayersWeights[layer] -= learningRate *
                (Network::NeuronsErrors[layer + 1] *
                 Network::NeuronsValues[layer].transpose());

                Network::LayersBiases[layer] -= learningRate *
                        Network::NeuronsErrors[layer + 1];
            }

            error += -(expected.array() * Network::NeuronsValues[Network::LayersCount - 1].array().log()).sum();
            correct += (pred == train_labels[i]);
        }

        uint32_t test_correct = 0;
        for (size_t i = 0; i < test_images.size(); ++i) {
            Network::NeuronsValues[0] = test_images[i];
            uint32_t pred = Network::ForwardFeed();
            test_correct += (pred == test_labels[i]);
        }

        std::cout << "Epoch " << epoch
                << " | Train Error: " << error / train_images.size()
                << " | Train Acc: " << static_cast<double>(correct) / static_cast<double>(train_images.size())
                << " | Test Acc: " << static_cast<double>(test_correct) / static_cast<double>(test_images.size())
                << std::endl;
        learningRate *= 0.95f;
    }


    std::ofstream weights_file("network_weights.txt");
    for (size_t i = 0; i < Network::LayersWeights.size(); ++i) {
        weights_file << Network::LayersWeights[i] << "\n";
        weights_file << Network::LayersBiases[i] << "\n";
    }

    while (true) {
        std::string filename;
        std::getline(std::cin, filename);
        if (filename == "exit") {
            return 0;
        }

        std::ifstream file(filename);
        Eigen::VectorXd input(property.NeuronsCount[0]);
        for (int i = 0; i < 784; ++i) {
            double temp;
            file >> temp;
            input(i) = temp;
        }

        Network::NeuronsValues[0] = input;

        uint32_t pred = Network::ForwardFeed();
        std::cout << "Pred: " << pred << std::endl;
    }
    return 0;
}
