from math import *
import numpy as np
import torch


####################################################################
#   One Point Spectra Data Generator 
#   (iso, shear: Kaimal, Simiu-Scanlan, Simiu-Yeo)
####################################################################

class OnePointSpectraDataGenerator:

    def __init__(self, **kwargs):
        self.DataPoints = kwargs.get('DataPoints', None)
        self.flow_type  = kwargs.get('flow_type', 'shear')  # 'shear', 'iso'
        self.data_type  = kwargs.get('data_type', 'Kaimal') # 'Kaimal', 'Simiu-Scanlan', 'Simiu-Yeo'

        if self.flow_type == 'iso':
            self.eval = self.eval_iso
        elif self.flow_type == 'shear':
            if self.data_type == 'Kaimal':
                self.eval = self.eval_shear_Kaimal
            elif self.data_type == 'SimiuScanlan':
                self.eval = self.eval_shear_SimiuScanlan
            elif self.data_type == 'SimiuYeo':
                self.eval = self.eval_shear_SimiuYeo
            elif self.data_type == 'iso':
                self.eval = self.eval_iso
            else:
                raise Exception()         
        else:
            raise Exception()

        if self.DataPoints is not None:
            self.generate_Data(self.DataPoints)

    def generate_Data(self, DataPoints):
        DataValues = np.zeros([len(DataPoints), 3, 3])
        for i, Point in enumerate(DataPoints):
            DataValues[i] = self.eval(*Point)
        self.Data = ( DataPoints, DataValues )
        self.Data = ( np.array(DataPoints), np.array(DataValues) )
        return self.Data


    #=============================================
    # Models
    #=============================================

    ### TODO: correct spectra ? off-diagonal ?

    def eval_iso(self, k1, z=1):
        C = 3.2
        L = 0.59
        F = np.zeros([3,3])
        F[0,0] = 9/55  * C / (L**(-2) + k1**2) **(5/6)
        F[1,1] = 3/110 * C * (3*L**(-2) + 8*k1**2) / (L**(-2) + k1**2) **(11/6)
        F[2,2] = 3/110 * C * (3*L**(-2) + 8*k1**2) / (L**(-2) + k1**2) **(11/6)
        return k1*F

    def eval_shear_Kaimal(self, k1, z=1):
        n = 1/(2*pi) * k1 * z
        F = np.zeros([3,3])
        F[0,0] = 52.5 * n / (1 + 33*n)**(5/3)
        F[1,1] = 8.5  * n / (1 + 9.5*n)**(5/3)
        F[2,2] = 1.05 * n / (1 + 5.3*n**(5/3))
        F[0,2] = -7  * n / (1 + 9.6*n)**(2.4)        
        return F

    def eval_shear_SimiuScanlan(self, k1, z=1):
        n = 1/(2*pi) * k1 * z
        F = np.zeros([3,3])
        F[0,0] = 100  * n / (1 + 50*n)**(5/3)
        F[1,1] = 7.5  * n / (1 + 9.5*n)**(5/3)
        F[2,2] = 1.68 * n / (1 + 10*n**(5/3))
        return F

    def eval_shear_SimiuYeo(self, k1, z=1):
        n = 1/(2*pi) * k1 * z
        F = np.zeros([3,3])
        F[0,0] = 100  * n / (1 + 50*n)**(5/3)
        F[1,1] = 7.5  * n / (1 + 10*n)**(5/3)
        F[2,2] = 1.68 * n / (1 + 10*n**(5/3))
        return F


####################################################################

####################################################################

class CoherenceDataGenerator:

    def __init__(self, **kwargs):
        self.DataGrids = kwargs.get('DataGrids', None)
        # if torch.is_tensor(self.DataPoints):
        #     self.DataPoints = self.DataPoints.cpu().detach().numpy()
        if self.DataGrids is not None:
            self.generate_Data(self.DataGrids)

    def generate_Data(self, DataGrids):
        d = len(DataGrids)
        DataPoints = np.stack(np.meshgrid(*DataGrids, indexing='ij'), axis=-1).reshape([-1,d])
        # DataValues = np.zeros(DataPoints.shape[0])
        DataValues = np.zeros([grid.size for grid in DataGrids])
        for i, Point in enumerate(DataPoints):
            DataValues.flat[i] = self.eval(*Point)
        self.Data = ( DataGrids, DataValues )
        return self.Data


    def eval(self, k1, y, z):        
        Vhub = 6
        Lc   = 8.1*42
        r    = np.sqrt(y**2+z**2)
        f    = k1
        x = (f*r/Vhub)**2 + (0.12*r/Lc)**2
        g = np.exp(-12*x**0.5)
        return g