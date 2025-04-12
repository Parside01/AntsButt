#include "Network.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

uint32_t Network::LayersCount = 0;
std::vector<Eigen::MatrixXd> Network::LayersWeights;
std::vector<Eigen::VectorXd> Network::LayersBiases;
std::vector<uint32_t> Network::NeuronsCount;
std::vector<Eigen::VectorXd> Network::NeuronsValues;
std::vector<Eigen::VectorXd> Network::NeuronsErrors;

double powTwo(double x) {
    return x * x;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double derivative_sigmoid(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu(double x) {
    return std::max(0.0, x);
}

double derivative_relu(double x) {
    return x > 0 ? 1.0 : 0.0;
}

void Network::Init(const NetworkProperty &prop) {
    LayersWeights.clear();
    LayersBiases.clear();
    NeuronsValues.clear();
    NeuronsErrors.clear();
    NeuronsCount.clear();

    LayersCount = prop.LayersCount;
    NeuronsCount = prop.NeuronsCount;

    for (uint32_t i = 0; i < LayersCount; ++i) {
        NeuronsValues.emplace_back(Eigen::VectorXd::Zero(NeuronsCount[i]));
        NeuronsErrors.emplace_back(Eigen::VectorXd::Zero(NeuronsCount[i]));
    }

    for (uint32_t i = 0; i < LayersCount - 1; ++i) {
        LayersWeights.emplace_back(Eigen::MatrixXd::Random(NeuronsCount[i + 1], NeuronsCount[i])
                           * sqrt(2.0 / NeuronsCount[i]));
        LayersBiases.emplace_back(Eigen::VectorXd::Zero(NeuronsCount[i + 1]));
    }
}

void Network::InitWidthFromFile(const std::string &filepath, const NetworkProperty &prop) {
    LayersCount = prop.LayersCount;
    NeuronsCount = prop.NeuronsCount;

    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filepath << std::endl;
        exit(EXIT_FAILURE);
    }
    LayersBiases.clear();
    LayersWeights.clear();
    NeuronsValues.clear();
    NeuronsErrors.clear();

    std::vector<double> allNumbers;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        double num;
        while (iss >> num) {
            allNumbers.push_back(num);
        }
    }


    size_t cursor = 0;

    for (uint32_t layer = 0; layer < LayersCount - 1; ++layer) {
        const uint32_t rows = NeuronsCount[layer + 1];
        const uint32_t cols = NeuronsCount[layer];

        Eigen::MatrixXd weights(rows, cols);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                weights(i, j) = allNumbers[cursor++];
            }
        }
        LayersWeights.push_back(weights);

        Eigen::VectorXd biases(rows);
        for (uint32_t i = 0; i < rows; ++i) {
            biases(i) = allNumbers[cursor++];
        }
        LayersBiases.push_back(biases);
    }

    if (cursor < allNumbers.size()) {
        std::cerr << "Warning: " << (allNumbers.size() - cursor)
                << " extra values in file" << std::endl;
    }

    for (uint32_t i = 0; i < LayersCount; ++i) {
        NeuronsValues.emplace_back(Eigen::VectorXd::Zero(NeuronsCount[i]));
        NeuronsErrors.emplace_back(Eigen::VectorXd::Zero(NeuronsCount[i]));
    }

    if (LayersWeights.size() != LayersCount - 1 || LayersBiases.size() != LayersCount - 1) {
        std::cerr << "Error: Loaded " << LayersWeights.size() << " weight matrices and "
                << LayersBiases.size() << " bias vectors, but need " << LayersCount - 1
                << " of each" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Network::InitWidthFromInput(const NetworkProperty& prop) {
    LayersCount = prop.LayersCount;
    NeuronsCount = prop.NeuronsCount;

    LayersBiases.clear();
    LayersWeights.clear();
    NeuronsValues.clear();
    NeuronsErrors.clear();

    size_t expected_count = 0;
    for (uint32_t i = 0; i < LayersCount - 1; ++i) {
        expected_count += NeuronsCount[i + 1] * NeuronsCount[i];
        expected_count += NeuronsCount[i + 1];
    }

    std::vector<double> allNumbers(expected_count);
    for (size_t i = 0; i < expected_count; ++i) {
        if (!(std::cin >> allNumbers[i])) {
            throw std::runtime_error("Error reading input");
        }
    }

    size_t cursor = 0;

    for (uint32_t layer = 0; layer < LayersCount - 1; ++layer) {
        const uint32_t rows = NeuronsCount[layer + 1];
        const uint32_t cols = NeuronsCount[layer];

        Eigen::MatrixXd weights(rows, cols);
        for (uint32_t i = 0; i < rows; ++i) {
            for (uint32_t j = 0; j < cols; ++j) {
                weights(i, j) = allNumbers[cursor++];
            }
        }
        LayersWeights.push_back(weights);

        Eigen::VectorXd biases(rows);
        for (uint32_t i = 0; i < rows; ++i) {
            biases(i) = allNumbers[cursor++];
        }
        LayersBiases.push_back(biases);
    }

    if (cursor < allNumbers.size()) {
        std::cerr << "Warning: " << (allNumbers.size() - cursor)
                  << " extra values in input" << std::endl;
    }

    for (uint32_t i = 0; i < LayersCount; ++i) {
        NeuronsValues.emplace_back(Eigen::VectorXd::Zero(NeuronsCount[i]));
        NeuronsErrors.emplace_back(Eigen::VectorXd::Zero(NeuronsCount[i]));
    }

    if (LayersWeights.size() != LayersCount - 1 || LayersBiases.size() != LayersCount - 1) {
        std::cerr << "Error: Loaded " << LayersWeights.size() << " weight matrices and "
                  << LayersBiases.size() << " bias vectors, but need " << LayersCount - 1
                  << " of each" << std::endl;
        exit(EXIT_FAILURE);
    }
}

uint32_t Network::ForwardFeed() {
    for (uint32_t i = 1; i < LayersCount; ++i) {
        Eigen::VectorXd nextLayerValues = LayersWeights[i - 1] * NeuronsValues[i - 1] + LayersBiases[i - 1];
        if (i == LayersCount - 1) {
            nextLayerValues = nextLayerValues.array() - nextLayerValues.maxCoeff();
            NeuronsValues[i] = nextLayerValues.array().exp() / nextLayerValues.array().exp().sum();
        } else {
            NeuronsValues[i] = nextLayerValues.unaryExpr(&relu);
        }
    }
    const Eigen::ArrayXd &lastLayerValues = NeuronsValues[LayersCount - 1];
    uint32_t pred;
    lastLayerValues.maxCoeff(&pred);
    return pred;
}

void Network::BackwardFeed(const Eigen::VectorXd &expected) {
    const uint32_t outputLayer = LayersCount - 1;

    NeuronsErrors[outputLayer] = (NeuronsValues[outputLayer] - expected);

    for (uint32_t i = outputLayer - 1; i > 0; --i) {
        NeuronsErrors[i] = (LayersWeights[i].transpose() * NeuronsErrors[i + 1]).cwiseProduct(
            NeuronsValues[i].unaryExpr(&derivative_relu)
        );
    }
}
