
import numpy as np
import torch
from scipy.special import hyp2f1

"""
==================================================================================================================
Von Karman energy spectrum (without scaling)
==================================================================================================================
"""

@torch.jit.script
def VKEnergySpectrum(kL):
    p  = 4
    cL = 1
    return kL**p / (cL+kL**2)**(5/6+p/2)


"""
==================================================================================================================
Mann's Eddy Liftime (numpy only - no torch)
==================================================================================================================
"""

def MannEddyLifetime(kL):
    x = kL.detach().numpy() if torch.is_tensor(kL) else kL
    y = x**(-2/3) / np.sqrt( hyp2f1(1/3, 17/6, 4/3, -x**(-2)) )
    y = torch.tensor(y, dtype=torch.float64) if torch.is_tensor(kL) else y
    return y

