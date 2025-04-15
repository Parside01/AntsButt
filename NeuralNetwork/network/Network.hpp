#ifndef NETWORK_HPP
#define NETWORK_HPP

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

struct NetworkProperty {
    uint32_t LayersCount;
    std::vector<uint32_t> NeuronsCount;
};

class Network {
public:
    static void Init(const NetworkProperty& prop);
    static void InitWidthFromFile(const std::string& filepath, const NetworkProperty &prop);
    static void InitWidthFromInput(const NetworkProperty &prop);
    static uint32_t ForwardFeed();
    static void BackwardFeed(const Eigen::VectorXd &expected);

    static uint32_t LayersCount;
    static std::vector<Eigen::MatrixXd> LayersWeights;

    static std::vector<uint32_t> NeuronsCount;
    static std::vector<Eigen::VectorXd> NeuronsValues;
    static std::vector<Eigen::VectorXd> NeuronsErrors;
    static std::vector<Eigen::VectorXd> LayersBiases;
};

#endif //NETWORK_HPP
