cmake_minimum_required(VERSION 3.8)

file(GLOB SOURCES "*.cpp")

add_executable(ckg-tool ${SOURCES})

target_link_libraries(ckg-tool PRIVATE cuda_kernel_gen)

target_include_directories(ckg-tool PRIVATE "../include")
