#include <stdio.h>
#include <cstring>


/////////////////////////////////////////////////////////////
// The class is a base class for query using pointers
// The () operator is set to be virtual in this class
// The dimension of the tensor should be specified in its subclass
template <typename T> class QueryBase {
public:
	QueryBase(T* _data): data(_data) {}
	__device__ virtual T& operator()(int) = 0;

protected:
	__device__ QueryBase(const QueryBase<T>&);
	T* data;
};

template <typename T> class ForwardQuery : public QueryBase<T> {
public:
	// Specified as 3 dimensional tensor
	ForwardQuery(T* _data, int _size)
				: QueryBase<T>(_data), size(_size) {}
	__device__ T& operator()(int num) { return this->data[num]; }

protected:
	int size;
};

template <typename T>
__global__ void hello(T* in, T* ks, T* out)
{
	__shared__ T tmp[10];
	tmp[threadIdx.x + 1] = in[threadIdx.x];
	if(threadIdx.x == 0) 
	{
		tmp[threadIdx.x] = 0;
		tmp[threadIdx.x + blockDim.x + 1] = 0;
	}
	__syncthreads();

	T ans = 0;
	for(int i=-1; i<=1; i++)
	{
		ans += (ks[i + 1] * tmp[threadIdx.x + i + 1]);
	}
	out[threadIdx.x] = ans;
	// __syncthreads();
}

template<typename T>
__global__ void testTemplate(QueryBase<T>& in, QueryBase<T>& out) {
	out(threadIdx.x) = in(threadIdx.x) + 1.0;
	// __syncthreads();
}

template <typename T>
__global__ void test2(T* in, T* out) {
	out[threadIdx.x] = in[threadIdx.x] + 1.0;
}

void showData(float* ms) {
	for(int i=0; i<8; i++) {
		printf("%f ", ms[i]);
	}
	printf("\n");
}

int main(void)
{
	//printf("Max shared memory: %d", CUpti_ActivityDevice::maxSharedMemoryPerBlock);
	int num_threads = 8;
	int num_blocks = 1;
	float data[8], kernels[3];
	for(int i=0; i<8; i++)
		data[i] = float(i) * 0.5;
	for(int i=0; i<3; i++)
		kernels[i] = 1.0;
	float outputs[8];

	float *in, *out, *ks;
	cudaMalloc((void**)&in, 32);
	cudaMalloc((void**)&out, 32);
	cudaMalloc((void**)&ks, 12);
	cudaMemcpy(in, data, 32, cudaMemcpyHostToDevice);
	cudaMemcpy(ks, kernels, 12, cudaMemcpyHostToDevice);
	ForwardQuery<float> d_in(in, 8), d_out(out, 8);

	hello<float><<< num_blocks, num_threads >>>(in, ks, out);
	cudaDeviceSynchronize();
	cudaMemcpy(outputs, out, 32, cudaMemcpyDeviceToHost);
	showData(outputs);
	printf("\n");

	cudaMemcpy(in, data, 32, cudaMemcpyHostToDevice);
	cudaMemcpy(ks, kernels, 12, cudaMemcpyHostToDevice);
	// testTemplate<float><<< num_blocks, num_threads >>>(d_in, d_out);
	test2<float><<<num_blocks, num_threads>>>(in, out);
	cudaDeviceSynchronize();
	cudaMemcpy(outputs, out, 32, cudaMemcpyDeviceToHost);
	showData(outputs);
	printf("\n");
	return 0;
}
