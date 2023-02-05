
import torch
import custom_ctc
import custom_ctc_gpu

import numpy as np

import torch.nn.functional as F
from torch.autograd import gradcheck

custom_ctc_f = custom_ctc.CustomCTCLossFunction.apply
custom_ctc_f_gpu = custom_ctc_gpu.CustomCTCLossFunction.apply

def test_ctc_loss_custom(device):
    batch_size = 64
    num_labels = 101
    target_length = 15
    gradcheck_input_size = 10

    ZERO_NONE = 0
    ZERO_SOME = 1
    ZERO_ALL = 2

    # input_length, vary_lengths, zero_lengths
    tests = [(150, False, ZERO_NONE),
                (150, True, ZERO_NONE),
                (50, True, ZERO_SOME),
                (50, True, ZERO_ALL)]

    tests += [(50, False, ZERO_NONE),
                (50, True, ZERO_NONE),
                (150, True, ZERO_SOME),
                (150, True, ZERO_ALL)]

    for input_length, vary_lengths, zero_mode in tests:
        targets = torch.randint(1, num_labels, (batch_size, target_length),
                                device=device, dtype=torch.long)
        x = torch.randn(gradcheck_input_size, dtype=torch.double, device=device, requires_grad=True)
        tile_factors = torch.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                    device=device)
        input_lengths = [(torch.randint(input_length // 2, input_length + 1, ()).item()
                            if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
        if zero_mode == ZERO_ALL:
            target_lengths = [0 for _ in range(batch_size)]
        else:
            target_lengths = [(torch.randint(target_length // 2, target_length + 1, ()).item()
                                if vary_lengths else target_length) for _ in range(batch_size)]
            if zero_mode == ZERO_SOME:
                idxes = torch.randint(0, batch_size, (10,))
                for i in idxes:
                    target_lengths[i] = 0

        num_realval = np.random.randint(1, 16)
        rv_x = torch.randn(gradcheck_input_size, dtype=torch.double, device=device, requires_grad=True)
        tile_factors_rv = torch.randn(batch_size * input_length * num_realval // gradcheck_input_size + 1,
                                    device=device)
                                    
        targets_realvals = torch.randn(batch_size, input_length, num_realval, dtype=torch.double)

        blank1 = np.random.randint(1, num_labels - 1)

        def ctc_after_softmax(x, rv):
            x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                        .view(batch_size, input_length, num_labels))
            rv_full = ((rv[:, None] * tile_factors_rv[None, :]).view(-1)[:input_length * batch_size * num_realval]
                        .view(batch_size, input_length, num_realval))
            log_probs = torch.log_softmax(x_full, 2)
            return custom_ctc_f(log_probs, targets, rv_full, targets_realvals, input_lengths, target_lengths, 2.2, 0, blank1, 'mean', True)

        gradcheck(ctc_after_softmax, [x, rv_x])


def test_ctc_loss_custom_gpu(device, fp = torch.float32):
    print('testing GPU gradient for %s' % str(fp))
    batch_size = 64
    num_labels = 101
    target_length = 15
    gradcheck_input_size = 10

    ZERO_NONE = 0
    ZERO_SOME = 1
    ZERO_ALL = 2

    # input_length, vary_lengths, zero_lengths
    tests = [(150, False, ZERO_NONE),
                (150, True, ZERO_NONE),
                (50, True, ZERO_SOME),
                (50, True, ZERO_ALL)]

    tests += [(50, False, ZERO_NONE),
                (50, True, ZERO_NONE),
                (150, True, ZERO_SOME),
                (150, True, ZERO_ALL)]

    for input_length, vary_lengths, zero_mode in tests:
        targets = torch.randint(1, num_labels, (batch_size, target_length),
                                device=device, dtype=torch.long)
        x = torch.randn(gradcheck_input_size, dtype=fp, device=device)
        tile_factors = torch.randn(input_length * batch_size * num_labels // gradcheck_input_size + 1,
                                    device=device)
        input_lengths = [(torch.randint(input_length // 2, input_length + 1, ()).item()
                            if vary_lengths or i == 0 else input_length) for i in range(batch_size)]
        if zero_mode == ZERO_ALL:
            target_lengths = [0 for _ in range(batch_size)]
        else:
            target_lengths = [(torch.randint(target_length // 2, target_length + 1, ()).item()
                                if vary_lengths else target_length) for _ in range(batch_size)]
            if zero_mode == ZERO_SOME:
                idxes = torch.randint(0, batch_size, (10,))
                for i in idxes:
                    target_lengths[i] = 0

        num_realval = np.random.randint(1, 16)
        rv_x = torch.randn(gradcheck_input_size, dtype=fp, device=device)
        tile_factors_rv = torch.randn(batch_size * input_length * num_realval // gradcheck_input_size + 1,
                                    device=device)
                                    
        targets_realvals = torch.randn(batch_size, input_length, num_realval, dtype=fp)

        blank1 = np.random.randint(1, num_labels - 1)

        x_full = ((x[:, None] * tile_factors[None, :]).view(-1)[:input_length * batch_size * num_labels]
                    .view(batch_size, input_length, num_labels))
        rv_full = ((rv_x[:, None] * tile_factors_rv[None, :]).view(-1)[:input_length * batch_size * num_realval]
                    .view(batch_size, input_length, num_realval))
        log_probs = torch.log_softmax(x_full, 2)
        log_probs.requires_grad_()
        rv_full.requires_grad_()
        grad_out = torch.randn(batch_size, device='cpu', dtype=fp)
        loss_native = custom_ctc_f(log_probs, targets, rv_full, targets_realvals, input_lengths, target_lengths, 1, 0, blank1, 'none', True)
        grad_native = torch.autograd.grad(loss_native, [log_probs, rv_full], grad_out)
        if torch.any(loss_native < 0) :
            breakpoint()
        
        log_probs.requires_grad_(False)
        rv_full.requires_grad_(False)
        log_probs = log_probs.cuda()
        rv_full = rv_full.cuda()
        log_probs.requires_grad_()
        rv_full.requires_grad_()
        targets = targets.cuda()
        targets_realvals = targets_realvals.cuda()

        loss_gpu = custom_ctc_f_gpu(log_probs, targets, rv_full, targets_realvals, input_lengths, target_lengths, 1, 0, blank1, 'none', True)
        grad_gpu = torch.autograd.grad(loss_gpu, [log_probs, rv_full], grad_out.cuda())
        #breakpoint()
        assert torch.allclose(loss_native, loss_gpu.cpu(), rtol=1e-4, atol=1e-4)
        print((grad_native[0] - grad_gpu[0].cpu()).abs().sum())
        if not torch.allclose(grad_native[0], grad_gpu[0].cpu(), rtol=1e-2, atol=1e-2) :
            breakpoint()
        print((grad_native[1] - grad_gpu[1].cpu()).abs().sum())
        assert torch.allclose(grad_native[1], grad_gpu[1].cpu(), rtol=1e-2, atol=1e-2)

if __name__ == '__main__' :
    test_ctc_loss_custom('cpu:0')
    for _ in range(100) :
        test_ctc_loss_custom_gpu('cpu:0')
        test_ctc_loss_custom_gpu('cpu:0', torch.double)
        #test_ctc_loss_custom_gpu('cpu:0', torch.half)
        print('test passed')
