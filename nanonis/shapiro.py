import matplotlib.pyplot as plt
import pickle
import time
import numpy as np
import useful as uf
from . import nanonis
import ast
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
spectra = nanonis.biasSpectroscopy()

def findmax(fnames):
    energy = []
    plt.figure()
    map = []
    for f in fnames:
        y = spectra.load(f)
        x, y = uf.data_smooth(spectra.bias,spectra.conductance,order=10)
        energy.append(x[y.argmax()])
        map.append(y)
    plt.imshow(map,extent=[spectra.bias.min(),spectra.bias.max(),0,1],aspect='auto',interpolation='nearest')
    return np.array(energy)


def dbm_mv(x):
    return 10**((x+47)/20) 


def mv_dbm(x):
    return -47+20*np.log10(x)


def TF_fast(fnames1,fnames2,p1,p2,TF_energy):
    ''' Outputs the correct power given two sets of spectras at 2 diferent powers
    
    Inputs are 2 lists of paths to the files .dat, 2 power arrays, and wanted energy of the outter coherence peak
    ''' 
    x0s = np.abs(sp.findmax(fnames1))
    x1s = np.abs(sp.findmax(fnames2))
    def line(x,m,q):
        return m*x+q

    def m_calc(x0,x1,p1,p2):
        return (p2-p1)/(x1-x0)

    def q_calc(p1,m,x0):
        return p1-m*x0

    new_P = []

    for i in range(10):
        m = m_calc(x0s[i],x1s[i],p1[i],p2[i])
        q = q_calc(p1[i],m,x0s[i])
        new_P.append(line(TF_energy,m,q))
    return new_P