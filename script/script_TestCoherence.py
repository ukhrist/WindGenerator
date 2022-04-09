
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import pickle
from math import log
import torch

from torch.nn import parameter

import sys
sys.path.append('./')
from source.SpectralCoherence import SpectralCoherence


####################################
### Configuration
####################################

config = {
    'type_EddyLifetime' :   'tauNet',  ### 'const', TwoThird', 'Mann', 'tauNet'
    'type_PowerSpectra' :   'RDT', ### 'RDT', 'zetaNet', 'C3Net', 'Corrector'
    'nlayers'           :   2,
    'hidden_layer_size' :   10,
    # 'nModes'            :   5, ### number of modes in the rational function in tauNet ### deprecated
    'learn_nu'          :   False, ### NOTE: Experiment 1: False, Experiment 2: True
    'plt_tau'           :   True,
    'tol'               :   1.e-3, ### not important
    'lr'                :   1,     ### learning rate
    'penalty'           :   1.e-1,
    'regularization'    :   1.e-1,
    'nepochs'           :   200,
    'curves'            :   [0,1,2,3],
    'data_type'         :   'Kaimal', ### 'Kaimal', 'SimiuScanlan', 'SimiuYeo', 'iso'
    'domain'            :   np.logspace(-1, 2, 20), ### NOTE: Experiment 1: np.logspace(-1, 2, 20), Experiment 2: np.logspace(-2, 2, 40)
    'noisy_data'        :   0*3.e-1, ### level of the data noise  ### NOTE: Experiment 1: zero, Experiment 2: non-zero
    'output_folder'     :   '/home/khristen/Projects/Brendan/2020_ontheflygenerator/code/data/',
    'fg_coherence'      :   True,
}
SpCoh = SpectralCoherence(**config)

### Data points
A = 50
N_f, N_y, N_z = 3, 3, 3
k1 = torch.logspace(-3,2, N_f, dtype=torch.float64)
Delta_y = torch.linspace(-A, A, N_y, dtype=torch.float64)
Delta_z = torch.linspace(-A, A, N_z, dtype=torch.float64)

### Forward run
y = SpCoh(k1, Delta_y, Delta_z)

pass

