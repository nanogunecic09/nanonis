# Superconductor analysis module
# nanoImaging group @ nanoGune

import numpy as np
from scipy import signal
from lmfit import Model
import nanonis
import matplotlib.pyplot as plt
import scipy.constants as const

# Evaluate the non-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array or value
#       * Delta is the superconducting gap at 0K


def BCS_curve(energy, Delta):
    Delta = np.complex(Delta)
    density = np.sign(energy)*np.real(np.divide(energy,
                                                np.sqrt(np.power(energy, 2)-np.power(Delta, 2))))
    return density

# Evaluate the Dynes-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array
#       * Delta is the superconducting gap at 0K
#       * dynesParameter is the Dynes factor


def dynes_curve(energy, Delta, dynesParameter):
    dynesParameter = np.complex(0, dynesParameter)
    density = np.sign(energy)*np.real(np.divide(energy-dynesParameter,
                                                np.sqrt(np.power(energy-dynesParameter, 2)-np.power(Delta, 2))))
    return density

# Evaluate the derivative Dynes-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array
#       * Delta is the superconducting gap at 0K
#       * dynesParameter is the Dynes factor


def dynes_curve_diff(energy, Delta, dynesParameter):
    dynesParameter = np.complex(0, dynesParameter)
    density_de = -np.sign(energy)*np.real(np.divide(np.square(Delta),
                                                    np.power(np.square(energy-dynesParameter)-np.square(Delta), 1.5)))
    density_de = np.nan_to_num(density_de)
    return density_de


# Fit a spectra with a dynes

def dynes_fit(fname, Delta=1e-3, dynes=1e-5):
    # initialize model
    dynes_Model = Model(dynes_curve)
    params = dynes_Model.make_params()
    params['Delta'].set(Delta, vary=True)
    params['dynesParameter'].set(dynes, vary=True)

    # load data
    spectra = nanonis.biasSpectroscopy()
    spectra.load(fname)
    x = np.array(spectra.bias)
    y = np.array(spectra.conductance)

    # perform fit
    result = dynes_Model.fit(y, energy=x, params=params)
    comps = result.eval(x=x)

    # visualization
    fig, ax = plt.subplots(1)
    ax.plot(x, y)
    ax.plot(x, comps)
    return result

# Fermi broadened dynes fit


class Superconductor():

    def __init__(self) -> None:
        pass

    def load(self,bias,conductance):
        self.bias = np.flip(bias)
        self.conductance = conductance

    def fdd(self, E, mu, T): #fermi Dirac function
        if T == 0:
            f = np.heaviside(-(E-mu), 1)
        else:
            f = 1/(1+np.exp((E-mu)/(const.k*T/const.e)))
        return f

    def dynesdos(self, E, Gamma, Delta): #dynes function
        dos = np.real((E-1j*Gamma)/np.sqrt((E-1j*Gamma)**2-Delta**2))
        return np.abs(dos)

    def dos(self,bias, T, Delta1, Gamma1, Delta2, Gamma2,A): #General calculation of the dos with dynes in the tip and sample, broadened by Fermi Dirac
        curr = []
        self.E = 10*np.linspace(-(Delta1+Delta2), (Delta1+Delta2), 1000)
        for Vp in self.bias:
            currp = np.trapz(self.dynesdos(self.E, Gamma1, Delta1)*self.dynesdos(self.E-Vp,
                             Gamma2, Delta2)*(self.fdd(self.E, Vp, T)-self.fdd(self.E, 0, T)), x=self.E)
            curr.append(currp)
        return A*np.gradient(np.array(curr))
    
    def fitModel(self, T, Delta1, Gamma1, Delta2, Gamma2,A):
        model = Model(self.dos)
        params = model.make_params()
        params['T'].set(T,vary=False)
        params['Delta1'].set(Delta1,vary=True)
        params['Gamma1'].set(Gamma1,min=1e-4,max=40e-6)
        params['Delta2'].set(Delta2,vary=False)
        params['Gamma2'].set(Gamma2,min=1e-4,max=40e-6)
        params['A'].set(A)

        self.fit_res = model.fit(self.conductance,params,bias=self.bias)
        self.fit_res_eval = self.fit_res.eval(x=self.bias)
        return 
    
    def showResults(self):
        fig,self.ax = plt.subplots(1)
        self.ax.plot(self.bias,self.conductance)
        self.ax.plot(self.bias,self.fit_res.eval(x=self.bias))

    def dec_Cut(self,Energy):
        idx_n = abs(self.bias+Energy).argmin()
        idx_p = abs(self.bias-Energy).argmin()
        self.bias = self.bias[idx_p:idx_n]
        self.conductance = self.conductance[idx_p:idx_n]
