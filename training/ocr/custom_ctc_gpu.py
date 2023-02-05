
import torch
import custom_ctc_cu

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
			input_lengths = Tensor(input_lengths).long().cpu()
		if isinstance(target_lengths, list) :
			target_lengths = Tensor(target_lengths).long().cpu()
		ctx.old_fp = log_probs.dtype
		# log_probs = log_probs.double()
		realval = realval.float()
		targets_realval = targets_realval.float()
		neg_log_likelihood, log_alpha = custom_ctc_cu.forward(log_probs, targets, realval, targets_realval, input_lengths, target_lengths, sigma, blank, blank1, zero_infinity)
		ctx.save_for_backward(neg_log_likelihood, log_alpha, log_probs, targets, realval, targets_realval, input_lengths, target_lengths)
		ctx.blank = blank
		ctx.blank1 = blank1
		ctx.zero_infinity = zero_infinity
		ctx.sigma = sigma
		ctx.reduction = reduction
		if reduction == 'mean' :
			ret = (neg_log_likelihood / target_lengths.to(log_probs.device).clamp_min(1)).mean()
		else :
			ret = neg_log_likelihood
		if torch.isnan(neg_log_likelihood).any() :
			print(neg_log_likelihood)
		# ret = ret.type(ctx.old_fp)
		ret = torch.nan_to_num(ret, nan = 0, posinf = 1, neginf = -1)
		return ret

	@staticmethod
	def backward(ctx, grad_out):
		neg_log_likelihood, log_alpha, log_probs, targets, realval, targets_realval, input_lengths, target_lengths = ctx.saved_tensors
		if ctx.reduction == 'mean' :
			if grad_out.numel() == 0 :
				grad_out = torch.ones_like(neg_log_likelihood)
			else :
				grad_out = grad_out.view(1).tile(neg_log_likelihood.size(0))
			grad_out /= target_lengths.to(log_probs.device).clamp_min(1)
			grad_out /= log_probs.size(0)
		# grad_out = grad_out.double()
		outputs_cls, outputs_realval = custom_ctc_cu.backward(grad_out, log_probs, targets, realval, targets_realval, input_lengths, target_lengths, neg_log_likelihood, log_alpha, ctx.sigma, ctx.blank, ctx.blank1, ctx.zero_infinity)
		
		if torch.isnan(outputs_cls).any() :
			print('warn outputs_cls NaN')
		if torch.isnan(outputs_realval).any() :
			print('warn outputs_realval NaN')
		# outputs_cls = outputs_cls.type(ctx.old_fp)
		outputs_cls = torch.nan_to_num(outputs_cls, nan = 0, posinf = 1, neginf = -1)
		# outputs_realval = outputs_realval.type(ctx.old_fp)
		outputs_realval = torch.nan_to_num(outputs_realval, nan = 0, posinf = 1, neginf = -1)
		
		return outputs_cls, None, outputs_realval, None, None, None, None, None, None, None, None

custom_ctc_loss = CustomCTCLossFunction.apply
