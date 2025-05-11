cmake_minimum_required (VERSION 3.8)

option(LINK_CUDA_KERNELS "Enable to link with generated kernels after ckg-tool run" FALSE)

project("cuda-kernel-gen-sample" LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(LINK_CUDA_KERNELS)
  file(GLOB KERNEL_SOURCES "generated/*.cu")

  add_library(kernels STATIC ${KERNEL_SOURCES})

  target_include_directories(kernels PUBLIC "generated" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  target_link_libraries(kernels PUBLIC ${CUDA_CUDART})

  link_libraries(kernels)
else()
  include_directories("../plugin" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  link_libraries(${CUDA_CUDART})
endif()

file(GLOB SOURCES "src/main.cpp")

add_executable(sample ${SOURCES})
