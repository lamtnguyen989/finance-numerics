#!/bin/bash

THREADS=$(nproc)  # This is the number of threads the computer has, basically this will pararellize the compilation across all cores

# Determines whether or not you want verbose compiler output (flags, debug info, etc.)
if [ "$1" = "1" ]; then
    VERBOSE_FLAG="VERBOSE=1"
else
    VERBOSE_FLAG=""
fi

# Clean 
rm black-scholes
#rm -rf CMakeFiles/
#rm CMakeCache.txt
#rm Makefile
make clean

# Main compilation and run command
cmake .
make -j$THREADS $VERBOSE_FLAG
./black-scholes

# Keep all of the outputs in a directory
DIR="output"
mkdir -p $DIR

BASE=("Black-Scholes-evolution" 
       "t=0.000000_cycle_1" 
       "t=0.000000_cycle_2")

EXTENSIONS=(".vtu"
            ".gnuplot")

for base in "${BASE[@]}"; do
    for ext in "${EXTENSIONS[@]}"; do
        #echo -e "${base}${ext}"
        if [ -f "${base}${ext}" ]; then
            mv "${base}${ext}" $DIR
        fi
    done
done

echo -e "All outputs moved to $DIR folder."

# Plot the evolution result
python3 plot-evolution.py