#include "kernels.cuh"
#include "kernel_wrappers.h"
void kernel_wrapper__0(int* in_1, float* in_2, float* out) {
	kernel__0<<<1,25>>>(in_1, in_2, out);
}
void kernel_wrapper__1(int* in_1_1, int* in_1_2, int* in_2_1_1, int* in_2_1_2, float* in_2_2, float* out) {
	kernel__1<<<1,25>>>(in_1_1, in_1_2, in_2_1_1, in_2_1_2, in_2_2, out);
}
void kernel_wrapper__2(int* in_1_1_1_1, int* in_1_1_1_2, int* in_1_1_2_1_1, int* in_1_1_2_1_2, float* in_1_1_2_2, int* in_1_2, int* in_2, float* out) {
	kernel__2<<<1,25>>>(in_1_1_1_1, in_1_1_1_2, in_1_1_2_1_1, in_1_1_2_1_2, in_1_1_2_2, in_1_2, in_2, out);
}
