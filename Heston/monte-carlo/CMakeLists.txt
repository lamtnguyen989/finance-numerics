cmake_minimum_required(VERSION 3.22)
project(hestonMC LANGUAGES CXX)

# Finding Kokkos
find_package(Kokkos REQUIRED)

# Sources
set(SRC_DIR ".")
set(SRC_FILE ${SRC_DIR}/Heston-MC.cc)

# Executables
add_executable(hestonMC ${SRC_FILE})
target_link_libraries(hestonMC Kokkos::kokkos)