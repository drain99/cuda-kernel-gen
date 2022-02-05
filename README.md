# CudaKernelGen

Takes a sequential program consisting of typical linear algebra operations and emits equivalent parallelized CUDA kernels.

## How it works
- A library is provided which uses C++ expression templates to encode computation at static time.
- User source is parsed and converted to Clang's AST format.
- Expression templates encoding the computation graphs is extracted from the AST.
- Extracted templates are parsed by the tool and equivalent parallelized CUDA kernels are generated from the computation graphs.
- A temporary copy of the provided library is modified by inserting the generated kernels at the appropiate positions.
- Unmodified user source is compiled and linked with the modified version of the library containing the kernel invocations.
