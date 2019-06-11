#include "..\include\MyKernels.h"
#include "..\include\MyKernelWrappers.h"

void kernel_wrapper__0(int* in_1_1, int* in_1_2, int* in_2_1, int* in_2_2,
					   int* out) {
  kernel__0<<<2, 50>>>(in_1_1, in_1_2, in_2_1, in_2_2, out);
}

