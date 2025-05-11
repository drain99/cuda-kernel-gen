cmake_minimum_required(VERSION 3.8)

file(GLOB SOURCES "*.cpp")

add_library(cuda_kernel_gen SHARED ${SOURCES})

target_include_directories(cuda_kernel_gen PUBLIC "../include" ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})

target_link_directories(cuda_kernel_gen PUBLIC ${LLVM_LIBRARY_DIRS})

target_link_libraries(cuda_kernel_gen PUBLIC LLVMCore LLVMSupport clangTooling clangBasic clangFrontend)
