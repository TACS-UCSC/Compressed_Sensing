import torch
import numpy as np

def rfft_abs_mirror_torch(data : torch.Tensor, axis):
    ## returns the rfft of the mirrored input along the axis given.
    data_mirror = torch.cat([data, torch.flip(data,[axis])],dim=axis)
    data_mirror_fft = torch.abs(torch.fft.rfft(data_mirror,dim=axis))
    return data_mirror_fft

