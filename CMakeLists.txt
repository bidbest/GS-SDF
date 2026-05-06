cmake_minimum_required(VERSION 3.0.2)
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()
project(neural_mapping LANGUAGES C CXX CUDA)

# Keep root-level builds working while the native source-of-truth lives in
# docker-build/ for Docker cache isolation.
include("${CMAKE_CURRENT_LIST_DIR}/docker-build/cmake/Build.cmake")
