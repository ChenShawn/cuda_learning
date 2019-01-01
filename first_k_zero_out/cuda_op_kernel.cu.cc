#if GOOGLE_CUDA  
#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void AddOneKernel(const T* in, const int N, T* out) {
	int idx = blockIdx.x * N + threadIdx.x;
	out[idx] = in[idx];
}

template <typename T>
void AddOneKernelLauncher(const T* in, const int batch_size, const int N, const int K, T* out) {
	AddOneKernel<T><<<batch_size, K>>>(in, N, out);
	cudaDeviceSynchronize();
}

template <typename Device, typename T>
class AddOneOp : public OpKernel {
public:
	explicit AddOneOp(OpKernelConstruction* context) : OpKernel(context) {
		OP_REQUIRES_OK(context, context->GetAttr("K", &K_));
	}

	void Compute(OpKernelContext* context) override {
		const Tensor& input_tensor = context->input(0);
		auto input = input_tensor.flat<T>();
		const int batch_size = input_tensor.shape().dim_size(0);
		const int N = input.size() / batch_size;

		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(), &output_tensor));
		auto output = output_tensor->flat<T>();
		OP_REQUIRES(context, K_>0 && K_ <= N,
			::tensorflow::errors::InvalidArgument("Invalid K value"));
		AddOneKernelLauncher<T>(input.data(), batch_size, N, K_, output.data());
	}
private:
	int K_;
};

REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU).TypeConstraint<int>("T"),
                        AddOneOp<GPUDevice, int>);
REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU).TypeConstraint<float>("T"),
                        AddOneOp<GPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("AddOne").Device(DEVICE_GPU).TypeConstraint<double>("T"),
                        AddOneOp<GPUDevice, double>);

#endif
