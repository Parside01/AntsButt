cmake_minimum_required(VERSION 3.29)
project(neural_network)

set(CMAKE_CXX_STANDARD 20)

option(CLIENT_MODE "Enable neural network training mode" OFF)

add_subdirectory(third_party/eigen)
add_library(neural_lib
        network/Network.hpp network/Network.cpp)

target_link_libraries(neural_lib Eigen3::Eigen)


#add_executable(${PROJECT_NAME} main-nn.cpp)
#add_executable(${PROJECT_NAME} main-learn-cli.cpp)
add_executable(${PROJECT_NAME} main-cli.cpp)
target_link_libraries(${PROJECT_NAME} neural_lib)
