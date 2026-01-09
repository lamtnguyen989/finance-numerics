#!/bin/bash

# Loading libraries with Spack
spack load cmake kokkos
BUILD_DIR=build

# Make sure FFT working (since Spack supported config is horrible at the time of writing this script)
mkdir -p tpls
cd tpls
git clone --recursive https://github.com/kokkos/kokkos-fft.git
cd ..

# Cleaning logs
rm -f run.log $BUILD_DIR/hestonFFT

# Cmake config
cmake   -B $BUILD_DIR \
        -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \

# Compile and run executable
make -C $BUILD_DIR -j$(nproc)
$BUILD_DIR/hestonFFT > run.log

# Result 
cat run.log