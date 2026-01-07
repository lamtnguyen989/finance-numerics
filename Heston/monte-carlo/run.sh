#!/bin/bash

# Loading libraries with Spack
spack load cmake kokkos

# Cleaning logs
rm -f run.log

# Cmake config
BUILD_DIR=build/
cmake   -B $BUILD_DIR \
        -DCMAKE_BUILD_TYPE=Release \
        -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON \

# Compile and run executable
make -C $BUILD_DIR -j$(nproc) -s
$BUILD_DIR/hestonMC > run.log

# Result 
cat run.log