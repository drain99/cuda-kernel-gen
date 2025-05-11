cmake_minimum_required(VERSION 3.8)

project("cuda-kernel-gen" LANGUAGES C CXX)

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory("lib")
add_subdirectory("tool")
