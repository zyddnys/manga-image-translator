"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import os
import argparse
import time


import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from .models import FFDNet
from .utils import normalize, variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
    
class FFDNetDenoiser:
    def __init__(self, _device, _sigma = 25, _weights_dir = 'denoising/models/', _in_ch = 3):
        self.sigma = _sigma / 255
        self.weights_dir = _weights_dir
        self.channels = _in_ch
        self.device = _device
        
        self.model = FFDNet(num_input_channels = _in_ch)
        self.load_weights()
        self.model.eval()
       
    
    def load_weights(self):
        weights_name = 'net_rgb.pth' if self.channels == 3 else 'net_gray.pth'
        weights_path = os.path.join(self.weights_dir, weights_name)
        if self.device == 'cuda':
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            device_ids = [0]
            self.model = nn.DataParallel(self.model, device_ids=device_ids).cuda()
        else:
            state_dict = torch.load(weights_path, map_location='cpu')
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_dataparallel_wrapper(state_dict)
        self.model.load_state_dict(state_dict)
        
    def get_denoised_image(self, imorig, sigma = None):
        
        if sigma is not None:
            cur_sigma = sigma / 255
        else:
            cur_sigma = self.sigma 
    
        if len(imorig.shape) < 3 or imorig.shape[2] == 1:
            imorig = np.repeat(np.expand_dims(imorig, 2), 3, 2)
            
        imorig = imorig[..., :3]

        if (max(imorig.shape[0], imorig.shape[1]) > 1200):
            ratio = max(imorig.shape[0], imorig.shape[1]) / 1200
            imorig = cv2.resize(imorig, (int(imorig.shape[1] / ratio), int(imorig.shape[0] / ratio)), interpolation = cv2.INTER_AREA)

        imorig = imorig.transpose(2, 0, 1)
 
        if (imorig.max() > 1.2):
            imorig = normalize(imorig)
        imorig = np.expand_dims(imorig, 0)

        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = imorig.shape
        if sh_im[2]%2 == 1:
            expanded_h = True
            imorig = np.concatenate((imorig, imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

        if sh_im[3]%2 == 1:
            expanded_w = True
            imorig = np.concatenate((imorig, imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)


        imorig = torch.Tensor(imorig)


        # Sets data type according to CPU or GPU modes
        if self.device == 'cuda':
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        imnoisy = imorig.clone()


        with torch.no_grad():
            imorig, imnoisy = imorig.type(dtype), imnoisy.type(dtype)
            nsigma = torch.FloatTensor([cur_sigma]).type(dtype)


        # Estimate noise and subtract it to the input image
        im_noise_estim = self.model(imnoisy, nsigma)
        outim = torch.clamp(imnoisy-im_noise_estim, 0., 1.)

        if expanded_h:
            imorig = imorig[:, :, :-1, :]
            outim = outim[:, :, :-1, :]
            imnoisy = imnoisy[:, :, :-1, :]

        if expanded_w:
            imorig = imorig[:, :, :, :-1]
            outim = outim[:, :, :, :-1]
            imnoisy = imnoisy[:, :, :, :-1]
        
        return variable_to_cv2_image(outim)
