#include "kernels.cuh"
__global__ 
void kernel__0(int* in_1, float* in_2, float* out) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 25)
		out[idx] = (in_1[idx]-in_2[idx]);
}
__global__ 
void kernel__1(int* in_1_1, int* in_1_2, int* in_2_1_1, int* in_2_1_2, float* in_2_2, float* out) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 25)
		out[idx] = ((in_1_1[idx]+in_1_2[idx])*((in_2_1_1[idx]+in_2_1_2[idx])-in_2_2[idx]));
}
__global__ 
void kernel__2(int* in_1_1_1_1, int* in_1_1_1_2, int* in_1_1_2_1_1, int* in_1_1_2_1_2, float* in_1_1_2_2, int* in_1_2, int* in_2, float* out) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < 25)
		out[idx] = ((((in_1_1_1_1[idx]+in_1_1_1_2[idx])*((in_1_1_2_1_1[idx]+in_1_1_2_1_2[idx])-in_1_1_2_2[idx]))+in_1_2[idx])-in_2[idx]);
}
