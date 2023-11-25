// Copyright (c) 2018 MathInf GmbH, Thomas Viehmann
// Modified by zyddnys
// Licensed under the BSD-3-Clause license
// This is the CPU implementation of the Connectionist Temporal Loss.
// We mostly follow Graves.
// 1. Graves et al: http://www.cs.toronto.edu/~graves/icml_2006.pdf
// Note from zyddnys:
//   Added regression capability to CTC loss, currently we use L2 regression, future L1 regression maybe added
//   Two BLANKS where BLANK is the BLANK in CTC, BLANK_1 means regression part of this target is ignored
// We use the equations from above link, but note that [1] has 1-based indexing and we (of course) use 0-based.
// Graves et al call the probabilities y, we use log_probs (also calling them inputs)
#include <torch/extension.h>

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Fill.h>

#include <numeric>
#include <type_traits>

// Functions that fill Tensors with constants.

#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/cpu/vec/vec.h>
#include <ATen/cpu/vec/functional.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cpu/Loops.h>

#include <ATen/native/Fill.h>

namespace at { namespace native {


template <typename scalar_t>
void fill_non_native_type(TensorIterator& iter, const Scalar& value_scalar) {
  auto value = value_scalar.to<scalar_t>().x;
  using H = typename std::make_signed<decltype(value)>::type;  // Signed type has more acceleration
  // Reserve the representation of value. static_cast<H>(value) is implementation defined.
  H val = *reinterpret_cast<H*>(std::addressof(value));
  cpu_kernel_vec</*check_dynamic_cast=*/false>(
      iter,
      [val]() -> H { return val; },
      [val]() { return Vectorized<H>(val); });
}

template <>
void fill_non_native_type<c10::complex<at::Half>>(TensorIterator& iter, const Scalar& value_scalar) {
  static_assert(sizeof(c10::complex<at::Half>) == sizeof(int32_t), "Size of ComplexHalf should be 32-bits");
  auto value = c10::complex<at::Half>(value_scalar.to<c10::complex<float>>());
  auto val = *reinterpret_cast<int32_t*>(std::addressof(value));
  cpu_kernel_vec</*check_dynamic_cast=*/false>(
      iter,
      [val]() -> int32_t { return val; },
      [val]() { return Vectorized<int32_t>(val); });
}

void fill_kernel(TensorIterator& iter, const Scalar& value_scalar) {
  if (iter.dtype() == ScalarType::Half) {
    fill_non_native_type<at::Half>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::BFloat16) {
    fill_non_native_type<at::BFloat16>(iter, value_scalar);
  } else if (iter.dtype() == ScalarType::ComplexHalf) {
    fill_non_native_type<c10::complex<at::Half>>(iter, value_scalar);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND(at::ScalarType::Bool, iter.dtype(), "fill_cpu", [&]() {
      scalar_t value = value_scalar.to<scalar_t>();
      cpu_kernel_vec(
          iter,
          [=]() -> scalar_t { return value; },
          [=]() { return Vectorized<scalar_t>(value); });
    });
  }
}



} // namespace native
} // namespace at

using namespace c10;
using namespace at;
using namespace at::native;

// this ad-hoc converts from targets (l in [1]) to augmented targets (l' in [1]) note that no bound-checking is done
template<typename target_t>
static inline int64_t get_target_prime(target_t* target, int64_t offset, int64_t stride, int64_t idx, int64_t BLANK) noexcept {
  if (idx % 2 == 0) {
    return BLANK;
  } else {
    return target[offset + stride * (idx / 2)];
  }
}

// log P(x|mu)
template<typename scalar_t>
scalar_t custom_distance_forward_log(scalar_t x, scalar_t mu, scalar_t sigma) noexcept {
  return -0.5 * std::log(2.0 * c10::pi<scalar_t>) - std::log(sigma) - 0.5 * (x - mu) * (x - mu) / (sigma * sigma);
}

// d(P(x|mu))/dmu
template<typename scalar_t>
scalar_t custom_distance_backward(scalar_t x, scalar_t mu, scalar_t sigma) noexcept {
  scalar_t val = 1.0 / (sigma * std::sqrt(2 * c10::pi<scalar_t>)) * std::exp(-0.5 * (x - mu) * (x - mu) / (sigma * sigma));
  return val * (x - mu) / (sigma * sigma);
}

// log P(x|mu)
template<typename scalar_t>
scalar_t custom_distance_forward_log_l1(scalar_t x, scalar_t mu, scalar_t sigma) noexcept {
  return - std::log(2 * sigma) - std::abs(x - mu) / sigma;
}


template<typename scalar_t>
scalar_t sgn(scalar_t v) noexcept {
  if (std::abs(v) < std::numeric_limits<scalar_t>::epsilon())
    return 0;
  return v / std::abs(v);
}

// d(P(x|mu))/dmu
template<typename scalar_t>
scalar_t custom_distance_backward_l1(scalar_t x, scalar_t mu, scalar_t sigma) noexcept {
  return -sgn(mu - x) * std::exp(-std::abs(x - mu) / sigma) / (2 * sigma * sigma);
}

#if 0
// d(log P(x|mu))/dmu
template<typename scalar_t>
scalar_t custom_distance_backward_log(scalar_t x, scalar_t mu) {
  return x - mu;
}

// P(x|mu)
template<typename scalar_t>
scalar_t custom_distance_forward(scalar_t x, scalar_t mu) {
  return 0;
}
#endif

// This kernel is a relatively straightforward implementation of the alpha calculation in the forward backward algorithm (section 4.1).
// A (minor) twist is that we are using log-calculations to enhance numerical stability (log_probs and log_alpha).
// The function returns the loss and the alphas, the alphas are kept for the backward step. The wrapper (ctc_loss below) hides
// the alphas from the user by only returning the loss.
template<typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor> custom_ctc_loss_cpu_template(
  const Tensor& log_probs,
  const Tensor& targets,
  const Tensor& realval,
  const Tensor& targets_realval,
  IntArrayRef input_lengths,
  IntArrayRef target_lengths,
  scalar_t const sigma,
  int64_t BLANK,
  int64_t BLANK_1
  ) {
  // log_probs: batch_size x input_len x num_labels
  // targets [int64]: batch_size x target_length OR sum(target_lengths)
  // realval [float]: batch_size x input_len x num_realval
  // targets_realval [float]: batch_size x max_target_length x num_realval
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  using target_t = typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;

  CheckedFrom c = "ctc_loss_cpu";
  auto log_probs_arg = TensorArg(log_probs, "log_probs", 1);
  auto targets_arg = TensorArg(targets, "targets", 2);
  auto realval_arg = TensorArg(realval, "realval", 3);
  auto targets_realval_arg = TensorArg(targets_realval, "targets_realval", 4);
  checkScalarType(c, targets_arg, target_scalar_type);
  checkDim(c, log_probs_arg, 3);
  checkDimRange(c, targets_arg, 1, 3);

  int64_t batch_size = log_probs.size(0);
  int64_t num_labels = log_probs.size(2);
  int64_t num_realval = realval.size(2);
  TORCH_CHECK((0 <= BLANK) && (BLANK < num_labels), "blank must be in label range");
  TORCH_CHECK((int64_t) input_lengths.size() == batch_size, "input_lengths must be of size batch_size");
  TORCH_CHECK((int64_t) target_lengths.size() == batch_size, "target_lengths must be of size batch_size");

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  size_t tg_target_stride;
  int64_t max_target_length = 0;
  std::vector<int64_t> tg_batch_offsets(batch_size);
  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
         max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
    checkSize(c, targets_arg, 0, pos);
  }
  else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets[i] = i * tg_batch_stride;
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(1);
    checkSize(c, targets_arg, 0, batch_size);
    TORCH_CHECK(targets.size(1) >= max_target_length,
             "Expected tensor to have size at least ", max_target_length, " at dimension 1, but got size ", targets.size(1), " for ", targets_arg,
             " (while checking arguments for ", c, ")");
  }

  int64_t max_input_length = log_probs.size(1);
  for (int64_t b = 0; b < batch_size; b++) {
    TORCH_CHECK(input_lengths[b] <= max_input_length,
             "Expected input_lengths to have value at most ", max_input_length, ", but got value ", input_lengths[b],
             " (while checking arguments for ", c, ")");
  }

  Tensor log_alpha = at::empty({batch_size, max_input_length, 2*max_target_length+1}, log_probs.options());
  Tensor neg_log_likelihood = at::empty({batch_size}, log_probs.options());

  auto log_probs_a_global = log_probs.accessor<scalar_t, 3>();
  auto log_alpha_a_global = log_alpha.accessor<scalar_t, 3>();
  auto targets_data = targets.data_ptr<target_t>();
  auto neg_log_likelihood_a = neg_log_likelihood.accessor<scalar_t, 1>();

  auto realval_data_a_global = realval.accessor<scalar_t, 3>();
  auto targets_realval_data_a_global = targets_realval.accessor<scalar_t, 3>();

  // alpha calculation for the first row, the three equations for alpha_1 above eq (6)
  // first the default
  log_alpha.narrow(1, 0, 1).fill_(neginf);
  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    for (int64_t b = start; b < end; b++) {
      int64_t input_length = input_lengths[b];
      int64_t target_length = target_lengths[b];
      auto log_probs_a = log_probs_a_global[b];
      auto log_alpha_a = log_alpha_a_global[b];
      int64_t tg_batch_offset = tg_batch_offsets[b];

      auto realval_data_a = realval_data_a_global[b];
      auto targets_realval_data_a = targets_realval_data_a_global[b];

      // the first two items of alpha_t above eq (6)
      log_alpha_a[0][0] = log_probs_a[0][BLANK];
      if (target_length > 0)
      {
        auto tgt = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 1, BLANK);
        scalar_t cur_logprob = log_probs_a[0][tgt];
        if (tgt != BLANK && tgt != BLANK_1)
        {
          for (int64_t i = 0; i < num_realval; ++i) {
            cur_logprob += custom_distance_forward_log(targets_realval_data_a[0][i], realval_data_a[0][i], sigma);
          }
        }
        log_alpha_a[0][1] = cur_logprob;
      }

      // now the loop over the inputs
      for (int64_t t=1; t<input_length; t++) {
        for (int64_t s=0; s<2*target_length+1; s++) {
          auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
          scalar_t cur_logprob = log_probs_a[t][current_target_prime];
          if (current_target_prime != BLANK && current_target_prime != BLANK_1) {
            for (int64_t i = 0; i < num_realval; ++i) {
              cur_logprob += custom_distance_forward_log(targets_realval_data_a[s / 2][i], realval_data_a[t][i], sigma);
            }
          }
          // this loop over s could be parallel/vectorized, too, but the required items are one index apart
          // alternatively, one might consider moving s to the outer loop to cache current_target_prime more (but then it needs to be descending)
          // for the cuda implementation, that gave a speed boost.
          // This is eq (6) and (7), la1,2,3 are the three summands. We keep track of the maximum for the logsumexp calculation.

          scalar_t la1 = log_alpha_a[t-1][s];
          scalar_t lamax = la1;
          scalar_t la2, la3;
          if (s > 0) {
            la2 = log_alpha_a[t-1][s-1];
            if (la2 > lamax)
              lamax = la2;
          } else {
            la2 = neginf;
          }
          if ((s > 1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s-2, BLANK) !=
                          current_target_prime)) {
            la3 = log_alpha_a[t-1][s-2];
            if (la3 > lamax)
              lamax = la3;
          } else {
            la3 = neginf;
          }
          if (lamax == neginf) // cannot do neginf-neginf
            lamax = 0;
          // this is the assignment of eq (6)
          log_alpha_a[t][s] = std::log(std::exp(la1-lamax)+std::exp(la2-lamax)+std::exp(la3-lamax))+lamax + cur_logprob;
        }
      }
      // the likelihood is the the sum of the last two alphas, eq (8), the loss is the negative log likelihood
      if (target_length == 0) {
        // if the target is empty then there is no preceding BLANK state and hence there is no path to merge
        neg_log_likelihood_a[b] = -log_alpha_a[input_length-1][0];
      } else {
        scalar_t l1 = log_alpha_a[input_length-1][target_length*2];
        scalar_t l2 = log_alpha_a[input_length-1][target_length*2-1];
        scalar_t m = std::max(l1, l2);
        m = ((m == neginf) ? 0 : m);
        scalar_t log_likelihood = std::log(std::exp(l1-m)+std::exp(l2-m))+m;
        neg_log_likelihood_a[b] = -log_likelihood;
      }
    }
  });

  return std::make_tuple(neg_log_likelihood, log_alpha);
}

// This is the backward. It consists of two phases:
// a) computing the beta analogous to the alphas in the forward (backward half of the forward-backward algorithm) (eq (10) and (11))
// b) collecting the per-activation characters for all s and wrapping the gradient (eq (16), the collection is the sum)
template<typename scalar_t, ScalarType target_scalar_type>
std::tuple<Tensor, Tensor> custom_ctc_loss_backward_cpu_template(
  const Tensor& grad_out,
  const Tensor& log_probs,
  const Tensor& targets,
  const Tensor& realval,
  const Tensor& targets_realval,
  IntArrayRef input_lengths,
  IntArrayRef target_lengths,
  const Tensor& neg_log_likelihood,
  const Tensor& log_alpha,
  scalar_t const sigma,
  int64_t BLANK,
  int64_t BLANK_1,
  bool zero_infinity
  ) {
  constexpr scalar_t neginf = -std::numeric_limits<scalar_t>::infinity();
  using target_t = typename std::conditional<target_scalar_type == kInt, int, int64_t>::type;
  TORCH_CHECK(log_probs.size(1) == realval.size(1),
             "Expected tensor of log_probs and realval to have the same size at dimension 1, but got size ", log_probs.size(1), " for log_probs and ", realval.size(1),
             " for realval");
  int64_t max_input_length = log_probs.size(1);
  int64_t batch_size = log_probs.size(0);
  int64_t num_labels = log_probs.size(2);
  int64_t num_realval = realval.size(2);
  Tensor grad = at::full_like(log_probs, neginf, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // at this point, this is log of empty sum
  Tensor grad_realval = at::full_like(realval, 0, LEGACY_CONTIGUOUS_MEMORY_FORMAT); // at this point, this is empty sum

  // The admin bits. We don't do much checking and assume that the forward did.
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t tg_target_stride;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  int64_t max_target_length;
  std::vector<int64_t> tg_batch_offsets(batch_size);

  if (targets.dim() == 1) { // concatenated targets
    int64_t pos = 0;
    max_target_length = 0;
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets[i] = pos;
      pos += target_lengths[i];
      if (max_target_length < target_lengths[i])
        max_target_length = target_lengths[i];
    }
    tg_target_stride = targets.stride(0);
  }
  else { // batch x max_target_length
    // dim is 2
    int64_t tg_batch_stride = targets.stride(0);
    for (int64_t i = 0; i < batch_size; i++) {
      tg_batch_offsets[i] = i * tg_batch_stride;
    }
    tg_target_stride = targets.stride(1);
    max_target_length = targets.size(1);
  }

  Tensor log_beta = at::empty_like(log_alpha, LEGACY_CONTIGUOUS_MEMORY_FORMAT);  // could be optimized to use only 2 rows
  auto log_probs_a_global = log_probs.accessor<scalar_t, 3>();
  auto log_alpha_a_global = log_alpha.accessor<scalar_t, 3>();
  auto log_beta_a_global = log_beta.accessor<scalar_t, 3>();
  auto grad_a_global = grad.accessor<scalar_t, 3>();
  auto grad_realval_a_global = grad_realval.accessor<scalar_t, 3>();
  auto targets_data = targets.data_ptr<target_t>();
  auto realval_a_global = realval.accessor<scalar_t, 3>();
  auto targets_realval_a_global = targets_realval.accessor<scalar_t, 3>();

  auto create_fill_iterator = [](const Tensor& tensor, IntArrayRef squash_dims) {
    return TensorIteratorConfig()
        .set_check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
        .check_all_same_dtype(false)
        .add_output(tensor)
        .resize_outputs(false)
        .declare_static_shape(tensor.sizes(), squash_dims)
        .build();
  };
  const auto fill_iter = create_fill_iterator(grad, /*squash_dims=*/1);
  const auto fill_1d_iter = create_fill_iterator(grad, /*squash_dims=*/{0, 1});
  const auto fill_log_beta_1d_iter = create_fill_iterator(log_beta, /*squash_dims=*/{0, 1});

  std::vector<scalar_t> grad_realval_acc(num_realval);

  at::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
    TensorIterator fill_iter_local(fill_iter);
    TensorIterator fill_1d_iter_local(fill_1d_iter);
    TensorIterator fill_log_beta_1d_iter_local(fill_log_beta_1d_iter);

    for (int64_t b = start; b < end; b++) {
      scalar_t gr = grad_out.accessor<scalar_t, 1>()[b];
      scalar_t nll = neg_log_likelihood.accessor<scalar_t, 1>()[b];
      auto grad_a = grad_a_global[b];
      auto grad_realval_a = grad_realval_a_global[b];
      if (zero_infinity && nll == std::numeric_limits<scalar_t>::infinity()) {
        // grad_batch.zero_();
        fill_iter_local.unsafe_replace_operand(0, grad_a.data());
        fill_kernel(fill_iter_local, 0);
        continue;
      }

      auto log_probs_a = log_probs_a_global[b];
      auto log_alpha_a = log_alpha_a_global[b];
      auto log_beta_a = log_beta_a_global[b];
      auto realval_data_a = realval_a_global[b];
      auto targets_realval_data_a = targets_realval_a_global[b];
      int64_t input_length = input_lengths[b];
      int64_t target_length = target_lengths[b];
      int64_t tg_batch_offset = tg_batch_offsets[b];

      // the initialization of beta before eq (10)
      // here we do the fill for each batch item separately, as the input lengths will differ, so the t in which
      // we start varies
      if (input_length > 0) {
        // log_beta.select(0, b).select(1, input_length-1).fill_(neginf);
        fill_log_beta_1d_iter_local.unsafe_replace_operand(
            0, log_beta_a[input_length - 1].data());
        fill_kernel(fill_log_beta_1d_iter_local, neginf);

        log_beta_a[input_length-1][2*target_length] = log_probs_a[input_length-1][BLANK];
        grad_a[input_length-1][BLANK] = log_alpha_a[input_length-1][2*target_length] + log_beta_a[input_length-1][2*target_length];

        if (target_length > 0) {
          auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, 2*target_length-1, BLANK);
          scalar_t cur_logprob = log_probs_a[input_length-1][current_target_prime];
          if (current_target_prime != BLANK && current_target_prime != BLANK_1) {
            for (int64_t i = 0; i != num_realval; ++i) {
              cur_logprob += custom_distance_forward_log(targets_realval_data_a[target_length-1][i], realval_data_a[input_length-1][i], sigma);
            }
          }
          log_beta_a[input_length-1][2*target_length-1] = cur_logprob;

          // the first two are a blank and a non-blank, so we know they are different and we don't need to do log+
          grad_a[input_length-1][current_target_prime] = log_alpha_a[input_length-1][2*target_length-1] + log_beta_a[input_length-1][2*target_length-1];
          if (current_target_prime != BLANK && current_target_prime != BLANK_1) {
            scalar_t log_prod_n = 0;
            for (int64_t i = 0; i != num_realval; ++i) {
              log_prod_n += custom_distance_forward_log(targets_realval_data_a[target_length-1][i], realval_data_a[input_length-1][i], sigma);
            }
            grad_a[input_length-1][current_target_prime] -= log_prod_n;
            scalar_t log_term1 = log_alpha_a[input_length-1][2*target_length-1] + log_beta_a[input_length-1][2*target_length-1] - log_probs_a[input_length-1][current_target_prime] - 2 * log_prod_n;
            for (int64_t i = 0; i != num_realval; ++i) {
              scalar_t log_constant_factors = log_prod_n - custom_distance_forward_log(targets_realval_data_a[target_length-1][i], realval_data_a[input_length-1][i], sigma);
              scalar_t grad_dp_dmu = std::exp(log_term1 + log_constant_factors + nll) * custom_distance_backward(targets_realval_data_a[target_length-1][i], realval_data_a[input_length-1][i], sigma);
              grad_realval_a[input_length-1][i] = -grad_dp_dmu * gr;
            }
          }
        }
      }

      // now loop applying eq (10) / (11)
      for (int64_t t=input_length-2; t>=0; t--) {
        // this loop over s could be parallel/vectorized and doesn't really need to be descending...
        // alternatively, one might consider moving s to the outer loop to cache current_target_prime more (but then it needs to be descending)
        // for the cuda implementation, that gave a speed boost.
        for (int64_t i = 0; i != num_realval; ++i)
          grad_realval_acc[i] = 0;
        for (int64_t s=2*target_length; s>=0; s--) {
          scalar_t lb1 = log_beta_a[t+1][s];
          scalar_t lbmax = lb1;
          scalar_t lb2, lb3;
          auto current_target_prime = get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s, BLANK);
          scalar_t cur_logprob = log_probs_a[t][current_target_prime];
          if (current_target_prime != BLANK && current_target_prime != BLANK_1) {
            for (int64_t i = 0; i < num_realval; ++i) {
              cur_logprob += custom_distance_forward_log(targets_realval_data_a[s / 2][i], realval_data_a[t][i], sigma);
            }
          }
          if (s < 2*target_length) {
            lb2 = log_beta_a[t+1][s+1];
            if (lb2 > lbmax)
              lbmax = lb2;
          } else {
            lb2 = neginf;
          }
          if ((s < 2*target_length-1) && (get_target_prime(targets_data, tg_batch_offset, tg_target_stride, s+2, BLANK) !=
                                          current_target_prime)) {
            lb3 = log_beta_a[t+1][s+2];
            if (lb3 > lbmax)
              lbmax = lb3;
          } else {
            lb3 = neginf;
          }
          if (lbmax == neginf)
            lbmax = 0;

          log_beta_a[t][s] = std::log(std::exp(lb1-lbmax)+std::exp(lb2-lbmax)+std::exp(lb3-lbmax))+lbmax + cur_logprob;
          // one might check whether one can vectorize this better when done after the t-loop...
          // now that we have beta, we fill in the sum of alpha*beta in eq (16)
          // in contrast to the cuda implementation, we only parallelize over the batch, so we don't have a concurrency
          // issue (several s can map to the same target character)
          // collected[b, t, target'[s]] "log+=" log_alpha[t, s]+log_beta[t, s]
          scalar_t log_alpha_beta =  log_alpha_a[t][s] + log_beta_a[t][s];
          scalar_t log_prod_n =  0;
          if (current_target_prime != BLANK && current_target_prime != BLANK_1) {
            for (int64_t i = 0; i < num_realval; ++i) {
              log_prod_n += custom_distance_forward_log(targets_realval_data_a[s / 2][i], realval_data_a[t][i], sigma);
            }
          }
          scalar_t log_alpha_beta_div_pr = log_alpha_beta - log_prod_n;
          scalar_t &lcab = grad_a[t][current_target_prime];
          if (lcab == neginf) {
            lcab = log_alpha_beta_div_pr;
          } else {
            scalar_t max = std::max(lcab, log_alpha_beta_div_pr);
            lcab = std::log(std::exp(lcab-max)+std::exp(log_alpha_beta_div_pr-max))+max;
          }
          if (current_target_prime != BLANK && current_target_prime != BLANK_1) {
            scalar_t log_term1 = log_alpha_beta - log_probs_a[t][current_target_prime] - 2 * log_prod_n;
            for (int64_t i = 0; i != num_realval; ++i) {
              scalar_t fl = custom_distance_forward_log(targets_realval_data_a[s / 2][i], realval_data_a[t][i], sigma);
              scalar_t log_constant_factors = log_prod_n - fl;
              scalar_t grad_dp_dmu = -std::exp(log_term1 + log_constant_factors + nll) * custom_distance_backward(targets_realval_data_a[s / 2][i], realval_data_a[t][i], sigma);
              grad_realval_acc[i] += grad_dp_dmu;
            }
            
          }
        }
        for (int64_t i = 0; i != num_realval; ++i) {
          grad_realval_a[t][i] = grad_realval_acc[i] * gr;
        }
      }

      // now grad has the sum of eq (16)
      // now we wrap up the calculation by adding in the remaining items of eq (16)
      // this could be a great target for further vectorization.
      // grad is the output gradient, nll is the loss. Note that the likelihood -nll is the Z of eq (16)
      for (int64_t t = 0; t < input_length; t++) { // or go for the full thing?
        for (int64_t c = 0; c < num_labels; c++) {
          scalar_t& res = grad_a[t][c];
          scalar_t lp = log_probs_a[t][c];
          res = (std::exp(lp)-std::exp(res + nll - lp)) * gr;
        }
      }

      // zero the remainder
      for (auto l : c10::irange(input_length, max_input_length)) {
        // grad_batch.select(0, l).zero_();
        fill_1d_iter_local.unsafe_replace_operand(0, grad_a[l].data());
        fill_kernel(fill_1d_iter_local, 0);
      }
    }
  });
  return std::make_tuple(grad, grad_realval);
}

std::tuple<Tensor, Tensor> custom_ctc_loss_cpu(
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
  (void)zero_infinity; // only used for backwards
  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_cpu", [&] {
      if (targets.scalar_type() == kLong) {
        return custom_ctc_loss_cpu_template<scalar_t, kLong>(log_probs, targets, realval, targets_realval, il, tl, static_cast<scalar_t>(sigma), BLANK, BLANK_1);
      } else {
        return custom_ctc_loss_cpu_template<scalar_t, kInt>(log_probs, targets, realval, targets_realval, il, tl, static_cast<scalar_t>(sigma), BLANK, BLANK_1);
      }
  });
}

std::tuple<Tensor, Tensor> custom_ctc_loss_backward_cpu(
  const Tensor& grad_out,
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
  Tensor ilc = input_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  Tensor tlc = target_lengths.to(Device(at::kCPU), at::kLong).contiguous();
  IntArrayRef il(ilc.data_ptr<int64_t>(), ilc.numel());
  IntArrayRef tl(tlc.data_ptr<int64_t>(), tlc.numel());
  return AT_DISPATCH_FLOATING_TYPES(log_probs.scalar_type(), "ctc_loss_backward_cpu", [&] {
      if (targets.scalar_type() == kLong) {
        return custom_ctc_loss_backward_cpu_template<scalar_t,kLong>(grad_out, log_probs, targets, realval, targets_realval, il, tl, neg_log_likelihood, log_alpha, static_cast<scalar_t>(sigma), BLANK, BLANK_1, zero_infinity);
      } else {
        return custom_ctc_loss_backward_cpu_template<scalar_t,kInt>(grad_out, log_probs, targets, realval, targets_realval, il, tl, neg_log_likelihood, log_alpha, static_cast<scalar_t>(sigma), BLANK, BLANK_1, zero_infinity);
      }
  });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &custom_ctc_loss_cpu, "custom CTC forward");
  m.def("backward", &custom_ctc_loss_backward_cpu, "custom CTC backward");
}
