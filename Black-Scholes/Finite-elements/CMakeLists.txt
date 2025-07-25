##
#  CMake script for the Finite Element code 
#  adapted from step-12 tutorial program
##

# Set the name of the project and target:
SET(TARGET "black-scholes")

SET(TARGET_SRC
  ${TARGET}.cc
)

# Minimum resources and versions
CMAKE_MINIMUM_REQUIRED(VERSION 4.0.1)

FIND_PACKAGE(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

# Error package handling stuff
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(${TARGET})
DEAL_II_INVOKE_AUTOPILOT()

# Optimization
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")