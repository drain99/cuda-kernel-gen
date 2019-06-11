#include "..\include\MyKernels.h"

__global__ void kernel__0(int *in_1_1, int *in_1_2, int *in_2_1, int *in_2_2,
                          int *out) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < 100)
    out[idx] = ((in_1_1[idx] + in_1_2[idx]) * (in_2_1[idx] + in_2_2[idx]));
}

