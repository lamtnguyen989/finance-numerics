cmake_minimum_required(VERSION 3.22)
project(bitReversal LANGUAGES CXX)

# Finding Kokkos
find_package(Kokkos REQUIRED)

# Sources
set(SRC_DIR "../src/")