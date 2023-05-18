"""
Different utilities such as orthogonalization of weights, initialization of
loggers, etc

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import cv2


def variable_to_cv2_image(varim):
    r"""Converts a torch.autograd.Variable to an OpenCV image

    Args:
        varim: a torch.autograd.Variable
    """
    nchannels = varim.size()[1]
    if nchannels == 1:
        res = (varim.data.cpu().numpy()[0, 0, :]*255.).clip(0, 255).astype(np.uint8)
    elif nchannels == 3:
        res = varim.data.cpu().numpy()[0]
        res = cv2.cvtColor(res.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
        res = (res*255.).clip(0, 255).astype(np.uint8)
    else:
        raise Exception('Number of color channels not supported')
    return res


def normalize(data):
    return np.float32(data/255.)

def remove_dataparallel_wrapper(state_dict):
    r"""Converts a DataParallel model to a normal one by removing the "module."
    wrapper in the module dictionary

    Args:
        state_dict: a torch.nn.DataParallel state dictionary
    """
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, vl in state_dict.items():
        name = k[7:] # remove 'module.' of DataParallel
        new_state_dict[name] = vl

    return new_state_dict

def is_rgb(im_path):
    r""" Returns True if the image in im_path is an RGB image
    """
    from skimage.io import imread
    rgb = False
    im = imread(im_path)
    if (len(im.shape) == 3):
        if not(np.allclose(im[...,0], im[...,1]) and np.allclose(im[...,2], im[...,1])):
            rgb = True
    print("rgb: {}".format(rgb))
    print("im shape: {}".format(im.shape))
    return rgb
