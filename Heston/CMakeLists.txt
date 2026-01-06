cmake_minimum_required(VERSION 3.22)
project(hestonFFT LANGUAGES CXX)

# Finding Kokkos
find_package(Kokkos REQUIRED)

# Adding dependecies
add_subdirectory(tpls/kokkos-fft)

# Sources
set(SRC_DIR "../src/")
set(SRC_FILE ${SRC_DIR}/Heston-FFT.cc)

# Executables
add_executable(hestonFFT ${SRC_FILE})
target_link_libraries(hestonFFT Kokkos::kokkos KokkosFFT::fft)