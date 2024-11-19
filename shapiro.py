import matplotlib.pyplot as plt
import pickle
import time
import numpy as np
import useful as uf
import nanonis
import ast
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d


def findmax(fnames):
    energy = []
    plt.figure()
    map = []
    for f in fnames:
        y = spectra.load(f)
        x, y = uf.data_smooth(spectra.bias,spectra.conductance,order=10)
        energy.append(x[y.argmax()])
        map.append(y)
    plt.imshow(map,aspect='auto',interpolation='nearest')
    return np.array(energy)


def dbm_mv(x):
    10**((x+47)/20)
    return 

def mv_dbm(x):
    return -47+20*np.log10(x)