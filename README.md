# cuda-kernel-gen

Provides a header-only tensor library `EtExpr.h` and the cuda kernel generation tool `ckg-tool` that automatically identifies and codegens fused operator kernels.
The codegen'd cuda kernels can be linked directly with the sources with no changes.

Advantages:
  - Avoid CPU overhead of multiple kernel dispatch & synchronization.
  - More device-side kernel optimization opportunities due to op fusion.

Dependencies:
  - CUDA toolkit (for compiling & linking cuda kernels)
  - LLVM & clang-dev (clang AST is used to identify expressions to generate kernels for)

## Sample C++ code using `EtExpr.h` library & `ckg-tool` codegen'd kernels

### C++ Source
```cpp
// create int & float tensors filled with initial data
Tensor<int, 1000> A(5), B(2);
Tensor<float, 1000> C(3);

// express computation lazily (no compute is done yet)
auto expr = (A - C) * B;

// .eval() forces a compute for which CUDA kernel is codegen'd
auto ret = expr.eval();
```

### Codegen'd kernel
```c
__global__ 
void kernel__0(int* in_1_1, float* in_1_2, int* in_2, float* out) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 1000)
		out[idx] = ((in_1_1[idx]-in_1_2[idx])*in_2[idx]);
}
```

### Codegen'd glue code
```cpp
  CUDA_KERNEL_GEN_ATTR OT eval() {
    OT result(0);
    if constexpr (std::is_same_v<std::decay_t<decltype(*this)>, ...>) {
      // call kernel wrapper on .eval()
      kernel_wrapper__0((*this).mExpr1.mExpr1.data(), (*this).mExpr1.mExpr2.data(), (*this).mExpr2.data(), result.data());
    }
    cudaDeviceSynchronize();
    return result;
  }
```


## Build & run sample

### Build `ckg-tool`
- `mkdir build && cd build`
- `cmake ..`
- `make`

### Add `ckg-tool` to `PATH`
- `export PATH=${PWD}/tool:${PATH}`

### Configure sample without codegen'd kernels
- `cd sample`
- `mkdir build && cd build`
- `cmake ..`

### Generate kernels with `ckg-tool`
- `ckg-tool` requires following arguments:
  - Plugin directory containing `EtExpr.h` header: `--plugin-dir ../../plugin`
  - Sample project root directory: `--root-dir ..`
  - Source files containing tensor ops: `../src/main.cpp`
  - `ckg-tool --plugin-dir ../../plugin --root-dir .. ../src/main.cpp`
  - CUDA kernels (sources & headers) are generated under `../generated`

### Reconfigure & build sample with codegen'd kernels
- `cmake -DLINK_CUDA_KERNELS=1 ..`
- `make`

### Run sample executble
- `./sample`
