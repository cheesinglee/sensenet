#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("point: float")
    .Input("split_indices: int32")
    .Input("split_values: float")
    .Input("left: int32")
    .Input("right: int32")
    .Output("output_index: int32");

class ZeroOutOp : public OpKernel {
 public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& point_tensor = context->input(0);
    const Tensor& sidx_tensor = context->input(1);
    const Tensor& sval_tensor = context->input(2);
    const Tensor& left_tensor = context->input(3);
    const Tensor& right_tensor = context->input(4);

    auto point = point_tensor.flat<float>();
    auto sidxs = sidx_tensor.flat<int32>();
    auto svals = sval_tensor.flat<float>();
    auto left = left_tensor.flat<int32>();
    auto right = right_tensor.flat<int32>();

    // Create an output tensor
    Tensor* ot = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({1}), &ot));
    auto output_flat = ot->flat<int32>();

    // Set all but the first element of the output tensor to 0.
    int current_idx = 0;
    int split_index = sidxs(0);

    while (split_index >= 0) {
        if (point(split_index) < svals(current_idx))
            current_idx = left(current_idx);
        else
            current_idx = right(current_idx);

        split_index = sidxs(current_idx);
    }

    output_flat(0) = point(0); // left(current_idx);
  }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);
