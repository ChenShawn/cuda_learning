#include <cstdio>

__global__ void helloFromGPU() {
	printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
	helloFromGPU<<< 1, 10 >>>();
	cudaDeviceSynchronize();
	return 0;
}

