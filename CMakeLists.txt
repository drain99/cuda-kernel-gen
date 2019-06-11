cmake_minimum_required (VERSION 3.8)

set(LLVM_DIR "C:\\Program Files\\LLVM\\lib64\\cmake\\llvm")
find_package(LLVM REQUIRED CONFIG)

project("CudaKernelGen")

include_directories("include")

add_subdirectory("src/ExprAST")
add_subdirectory("src/KernelGen")
add_subdirectory("src/TemplateParser")
add_subdirectory("src/KernelWriter")
add_subdirectory("src/Tool")
