cmake_minimum_required (VERSION 3.8)

# set llvm_dir manually.
set(LLVM_DIR "D:\\LLVM(modified)\\install\\lib64\\cmake\\llvm")
find_package(LLVM REQUIRED CONFIG)

project("CudaKernelGen")

include_directories("include")

add_subdirectory("src/ExprAST")
add_subdirectory("src/KernelGen")
add_subdirectory("src/KernelWriter")
add_subdirectory("src/SourceParser")
add_subdirectory("src/Tool")
