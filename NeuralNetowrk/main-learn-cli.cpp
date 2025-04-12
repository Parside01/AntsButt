#include <iostream>
#include <fstream>
#include "network/Network.hpp"

int main(int argc, char **argv) {
    NetworkProperty property;
    property.LayersCount = 3;
    property.NeuronsCount = {784, 256, 10};

    Network::InitWidthFromFile("C:\\Drafts\\github.com\\neural_network\\cmake-build-release\\network_weights.txt", property);

    std::vector<Eigen::VectorXd> memory;
    std::vector<Eigen::VectorXd> memoryExpectArr;

    while (true) {
        char start;
        std::cout << "Enter start: [y/n] ";
        std::cin.clear();
        std::cin >> start;
        if (start == 'n') break;

        std::ifstream file("C:/Users/vadim/GolandProjects/neural-network-cli/image.txt");
        Eigen::VectorXd input(property.NeuronsCount[0]);
        for (int i = 0; i < 784; ++i) {
            double temp;
            file >> temp;
            input(i) = temp;
        }

        Network::NeuronsValues[0] = input;

        uint32_t pred = Network::ForwardFeed();
        std::cout << "Predict: " << pred << std::endl;

        int32_t expect;
        std::cout << "Enter expect value: ";
        std::cin.clear();
        std::cin >> expect;

        if (expect == -1) continue;

        double learningRate = 0.001f;
        Eigen::VectorXd expected = Eigen::VectorXd::Zero(10);
        expected[expect] = 1.0;

        memory.push_back(input);
        memoryExpectArr.push_back(expected);

        Network::BackwardFeed(expected);
        for (int i = 0; i <= 50; i++) {
            for (uint32_t j = 0; j < memory.size(); j++) {
                Network::NeuronsValues[0] = memory[j];
                Network::ForwardFeed();
                Eigen::VectorXd memoryExpected = memoryExpectArr[j];
                Network::BackwardFeed(memoryExpected);

                for (uint32_t layer = 0; layer < Network::LayersCount - 1; ++layer) {
                    Network::LayersWeights[layer] -= learningRate *
                    (Network::NeuronsErrors[layer + 1] *
                     Network::NeuronsValues[layer].transpose());

                    Network::LayersBiases[layer] -= learningRate *
                            Network::NeuronsErrors[layer + 1];
                }
            }
        }
    }

    std::ofstream weights_file("network_weights.txt");
    for (size_t i = 0; i < Network::LayersWeights.size(); ++i) {
        weights_file << Network::LayersWeights[i] << "\n";
        weights_file << Network::LayersBiases[i] << "\n";
    }

    return 0;
}
