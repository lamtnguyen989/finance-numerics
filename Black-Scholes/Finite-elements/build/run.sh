#!/bin/bash

CORES=$(nproc)  # This is the number of cores the computer has, basically this will pararellize the compilation across all cores

# Determines whether or not you want verbose compiler output (flags, debug info, etc.)
if [$1 == "1"]; then
    VERBOSE_FLAG="VERBOSE=1"
else
    VERBOSE_FLAG=""
fi

# Clean 
rm black-scholes
rm -rf CMakeFiles/
rm CMakeCache.txt
rm Makefile

# Main compilation and run command
cmake .
make -j$CORES $VERBOSE_FLAG
./black-scholes