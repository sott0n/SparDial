cmake_minimum_required(VERSION 3.13.4)
project(spardial VERSION 1.0 LANGUAGES CXX C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(SPARDIAL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SPARDIAL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(FATAL_ERROR "SparDial requires in-tree build with LLVM_EXTERNAL_PROJECTS")
else()
    message(STATUS "SparDial in-tree build.")
    set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
    set(MLIR_INCLUDE_DIR ${LLVM_MAIN_SRC_DIR}/../mlir/include)
    set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
    set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/externals/torch-mlir/include)

list(APPEND CMAKE_MODULE_PATH ${MLIR_MAIN_SRC_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${LLVM_MAIN_SRC_DIR}/cmake)

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_subdirectory(include)
add_subdirectory(lib)

set(SPARDIAL_PYTHON_PACKAGES_DIR "${SPARDIAL_BINARY_DIR}/python_packages")
add_subdirectory(python)

# Add tests
add_subdirectory(tests)

message(STATUS "SparDial configuration complete")
