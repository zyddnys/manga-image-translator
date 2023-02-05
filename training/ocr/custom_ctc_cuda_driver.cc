
#include <torch/extension.h>

#include <vector>

using torch::Tensor;
using torch::IntArrayRef;

std::tuple<Tensor, Tensor> custom_ctc_loss_gpu(
  const Tensor& log_probs,
  const Tensor& targets,
  const Tensor& realval,
  const Tensor& targets_realval,
  IntArrayRef input_lengths,
  IntArrayRef target_lengths,
  double const sigma,
  int64_t BLANK,
  int64_t BLANK_1
);
std::tuple<Tensor, Tensor> custom_ctc_loss_backward_gpu(
  const Tensor& grad,
  const Tensor& log_probs,
  const Tensor& targets,
  const Tensor& realval,
  const Tensor& targets_realval,
  IntArrayRef input_lengths,
  IntArrayRef target_lengths,
  const Tensor& neg_log_likelihood,
  const Tensor& log_alpha,
  double const sigma,
  int64_t BLANK,
  int64_t BLANK_1,
  bool zero_infinity
);


std::tuple<Tensor, Tensor> custom_ctc_loss_gpu_driver(
  const Tensor& log_probs,
  const Tensor& targets, 
  const Tensor& realval,
  const Tensor& targets_realval,
  const Tensor& input_lengths,
  const Tensor& target_lengths,
  double const sigma,
  int64_t BLANK,
  int64_t BLANK_1,
  bool zero_infinity
) {
  (void)zero_infinity;
  Tensor ilc = input_lengths.contiguous();
  Tensor tlc = target_lengths.contiguous();
  IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  return custom_ctc_loss_gpu(log_probs, targets, realval, targets_realval, il, tl, sigma, BLANK, BLANK_1);
}

std::tuple<Tensor, Tensor> custom_ctc_loss_backward_gpu_driver(
  const Tensor& grad,
  const Tensor& log_probs,
  const Tensor& targets,
  const Tensor& realval,
  const Tensor& targets_realval,
  const Tensor& input_lengths,
  const Tensor& target_lengths,
  const Tensor& neg_log_likelihood,
  const Tensor& log_alpha,
  double const sigma,
  int64_t BLANK,
  int64_t BLANK_1,
  bool zero_infinity
) {
  Tensor ilc = input_lengths.contiguous();
  Tensor tlc = target_lengths.contiguous();
  IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  return custom_ctc_loss_backward_gpu(grad, log_probs, targets, realval, targets_realval, il, tl, neg_log_likelihood, log_alpha, sigma, BLANK, BLANK_1, zero_infinity);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &custom_ctc_loss_gpu_driver, "custom CTC forward (CUDA)");
  m.def("backward", &custom_ctc_loss_backward_gpu_driver, "custom CTC backward (CUDA)");
}
