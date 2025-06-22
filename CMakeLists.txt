cmake_minimum_required(VERSION 3.15)
project(MLIR_Tutorial LANGUAGES C CXX)

if(POLICY CMP0116)
    cmake_policy(SET CMP0116 NEW)
endif()
if(POLICY CMP0114)
    cmake_policy(SET CMP0114 NEW)
endif()

set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING " " FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-fPIC)
add_compile_options(-fno-rtti)

set(LLVM_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/externals/llvm-project/build")

if(NOT EXISTS "${LLVM_BUILD_DIR}")
    message(FATAL_ERROR "LLVM build directory not found: ${LLVM_BUILD_DIR}")
endif()

set(LLVM_DIR "${LLVM_BUILD_DIR}/lib/cmake/llvm")
set(MLIR_DIR "${LLVM_BUILD_DIR}/lib/cmake/mlir")

if(NOT EXISTS "${LLVM_DIR}/LLVMConfig.cmake")
    message(FATAL_ERROR "LLVMConfig.cmake not found in: ${LLVM_DIR}")
endif()

if(NOT EXISTS "${MLIR_DIR}/MLIRConfig.cmake")
    message(FATAL_ERROR "MLIRConfig.cmake not found in: ${MLIR_DIR}")
endif()

list(APPEND CMAKE_MODULE_PATH ${LLVM_DIR})
list(APPEND CMAKE_MODULE_PATH ${MLIR_DIR})

include(LLVMConfig)
include(MLIRConfig)
include(TableGen)
include(AddLLVM)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
