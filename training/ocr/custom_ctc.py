
import torch
import custom_ctc_cpp

Tensor = torch.Tensor

class CustomCTCLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        log_probs: Tensor,
        targets: Tensor,
        realval: Tensor,
        targets_realval: Tensor,
        input_lengths: Tensor,
        target_lengths: Tensor,
        sigma: float = 1,
        blank: int = 0,
        blank1: int = 0,
        reduction: str = "mean",
        zero_infinity: bool = False
        ):
        assert reduction in ['none', 'mean']
        if isinstance(input_lengths, list) :
            input_lengths = Tensor(input_lengths).long().to(log_probs.device)
        if isinstance(target_lengths, list) :
            target_lengths = Tensor(target_lengths).long().to(log_probs.device)
        neg_log_likelihood, log_alpha = custom_ctc_cpp.forward(log_probs, targets, realval, targets_realval, input_lengths, target_lengths, sigma, blank, blank1, zero_infinity)
        ctx.save_for_backward(neg_log_likelihood, log_alpha, log_probs, targets, realval, targets_realval, input_lengths, target_lengths)
        ctx.blank = blank
        ctx.blank1 = blank1
        ctx.zero_infinity = zero_infinity
        ctx.sigma = sigma
        ctx.reduction = reduction
        if reduction == 'mean' :
            return (neg_log_likelihood / target_lengths.clamp_min(1)).mean()
        return neg_log_likelihood

    @staticmethod
    def backward(ctx, grad_out):
        neg_log_likelihood, log_alpha, log_probs, targets, realval, targets_realval, input_lengths, target_lengths = ctx.saved_tensors
        if ctx.reduction == 'mean' :
            if grad_out.numel() == 0 :
                grad_out = torch.ones_like(neg_log_likelihood)
            else :
                grad_out = grad_out.view(1).tile(neg_log_likelihood.size(0))
            grad_out /= target_lengths.clamp_min(1)
            grad_out /= log_probs.size(0)
        outputs_cls, outputs_realval = custom_ctc_cpp.backward(grad_out, log_probs, targets, realval, targets_realval, input_lengths, target_lengths, neg_log_likelihood, log_alpha, ctx.sigma, ctx.blank, ctx.blank1, ctx.zero_infinity)
        return outputs_cls, None, outputs_realval, None, None, None, None, None, None, None, None
