#include <cstdio>
template<typename T>
__device__ __inline__ void add(T& val) {
	val += 1;
}

template<typename T>
__global__ void func(T* ptr) {
	add<T>(ptr[blockIdx.x]);
}

int main() {
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	int *h_ptr, *d_ptr;
	cudaHostAlloc(&h_ptr, 20, cudaHostAllocDefault);
	for(int i=0; i<5; i++) {
		h_ptr[i] = i*2 + 1;
	}
	cudaMalloc((void**)&d_ptr, 20);
	cudaMemcpyAsync(d_ptr, h_ptr, 20, cudaMemcpyHostToDevice, stream);
	func<<<5, 1, 0, stream>>>(d_ptr);
	cudaMemcpyAsync(h_ptr, d_ptr, 20, cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	for(int i=0; i<5; i++) {
		printf("%d ", h_ptr[i]);
	}
	printf("\n");
	return 0;
}


