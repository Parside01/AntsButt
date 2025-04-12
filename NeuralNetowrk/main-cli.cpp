#include <iostream>
#include <fstream>
#include "network/Network.hpp"

int main(int argc, char **argv) {
    NetworkProperty property;
    property.LayersCount = 3;
    property.NeuronsCount = {784, 256, 10};

    Network::InitWidthFromInput(property);

    Eigen::VectorXd input(property.NeuronsCount[0]);
    std::cerr << "Reading " << property.NeuronsCount[0] << " pixel values from stdin..." << std::endl;

    for (int i = 0; i < property.NeuronsCount[0]; ++i) {
        double temp;
        if (!(std::cin >> temp)) {
            std::cerr << "Error reading pixel value " << i << " from stdin." << std::endl;
            return 1;
        }
        input(i) = temp;
    }
    std::cerr << "Finished reading pixel values." << std::endl;

    Network::NeuronsValues[0] = input;

    uint32_t pred = Network::ForwardFeed();

    std::cout << pred << std::endl;

    return 0;
}
