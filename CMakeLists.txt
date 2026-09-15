cmake_minimum_required(VERSION 3.15)
project(skytouch_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11 (provided by scikit-build-core / pip environment)
find_package(pybind11 CONFIG REQUIRED)

# Include header directory
include_directories(include)

# Define the pybind11 module
# The module name '_core' will be imported in Python as 'skytouch._core'
pybind11_add_module(_core src/main.cpp)

# Install rule required by scikit-build-core
# Install target into the skytouch module directory
install(TARGETS _core DESTINATION skytouch)