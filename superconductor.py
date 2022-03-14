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

    def load(self,bias,conductance,Erange,Epx):
        self.biasHR = np.linspace(-Erange*1e-3,Erange*1e-3,Epx)
        self.conductance = conductance
        self.bias = bias

    def fdd(self, E, mu, T): #fermi Dirac function
        if T == 0:
            f = np.heaviside(-(E-mu), 1)
        else:
            f = 1/(1+np.exp((E-mu)/(const.k*T/const.e)))
        return f

    def dynesdos(self, E, Gamma, Delta): #dynes function
        dos = np.real((E-1j*Gamma)/np.sqrt((E-1j*Gamma)**2-Delta**2))
        return np.abs(dos)/np.abs(dos)[0]

    def arnolddos(self,E,x=1,k=1.2e10,d=3.8e-7,Delta=0.75e-3,gamma=0.0000001):
        #atomic units constants
        a0 = 5.2917721e-11
        m_e = 0.510e6/const.c**2
        Ry_eV = const.hbar**2/(const.e**2*m_e*a0**2)
        #convert into atomic units
        Delta = Delta/Ry_eV
        d = d/a0
        k = k*a0
        x = -d
        #generate energy array
        E = E/Ry_eV +gamma*1j
        #green function compute
        m_e = 1
        kp = np.sqrt(k**2 + 2*m_e*E)
        km = np.sqrt(k**2  -2*m_e*E)
        FE = np.abs(E)/(np.sqrt(E**2-Delta**2))
        G =(m_e/(k))*((1j* FE* np.cos(kp*x+km*d) - np.sin(kp*x+km*d) )*( np.cos(kp*(x+d) ))/((1j*FE* np.sin((kp-km)*d)-np.cos((kp-km)*d)) ))
        return np.abs(np.imag(G))



    def dos(self,bias, T, Delta1, Gamma1, Delta2, Gamma2,A): #General calculation of the dos with dynes in the tip and sample, broadened by Fermi Dirac
        curr = []
        self.E = 10*np.linspace(-2*(Delta1+Delta2), 2*(Delta1+Delta2), 10000)
        for Vp in self.bias:
            currp = np.trapz(self.dynesdos(self.E, Gamma1, Delta1)*self.dynesdos(self.E-Vp,
                             Gamma2, Delta2)*(self.fdd(self.E, Vp, T)-self.fdd(self.E, 0, T)), x=self.E)
            curr.append(currp)
        return A*np.gradient(np.array(curr))
    

    def dynesconv(self,bias,T,Delta,Gamma):
        curr = []
        for Vp in self.biasHR:
            currp = np.trapz(self.dynesdos(self.biasHR, Gamma, Delta)*(self.conductance/self.conductance[0])*(self.fdd(self.biasHR, Vp, T)-self.fdd(self.biasHR, 0, T)),x=self.biasHR)
            curr.append(currp)
        return np.gradient(np.array(curr))

    def dynes_arnold_conv(self,x,T,Delta,Gamma,gamma=1,d=1,k=1.2e10):
        curr = []
        self.E = np.linspace(-10*(Delta), 10*(Delta), 2000)
        for Vp in self.bias:
            currp = np.trapz(self.dynesdos(self.E, Gamma, Delta)*(self.arnolddos(self.E-Vp,gamma=gamma,d=d,k=k))*(self.fdd(self.E, Vp, T)-self.fdd(self.E, 0, T)),x=self.E)
            curr.append(currp)
        out = -np.gradient(np.array(curr))
        return out/out[0]

    def cond_spline():
        pass

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
    
    def fit_arnold(self):
        model = Model(self.dynes_arnold_conv)
        self.params = model.make_params()
        self.params['T'].set(1,vary=False)
        self.params['Delta'].set(0.31e-3,vary=False)
        self.params['Gamma'].set(20e-6,min=0,vary=False)
        self.params['gamma'].set(5e-6,min=0)
        self.params['d'].set(3.8e-7)
        self.params['k'].set(3.2e9,vary=False)

        self.fit_res = model.fit(self.conductance,self.params,x=self.bias)
        self.fit_res_eval = self.fit_res.eval(x=self.bias)


    def showResults(self):
        fig,self.ax = plt.subplots(1)
        self.ax.plot(self.bias,self.conductance)
        self.ax.plot(self.bias,self.fit_res.eval(x=self.bias))

    def dec_Cut(self,Energy):
        idx_n = abs(self.bias+Energy).argmin()
        idx_p = abs(self.bias-Energy).argmin()
        self.bias = self.bias[idx_p:idx_n]
        self.conductance = self.conductance[idx_p:idx_n]



class Arnold(): #compute Arnold model DOS in atomic units

    def __init__(self):
        pass

    def G(self,x=1,Erange=1,k=1,d=1,Delta=1,gamma=1,Epx=10000):
        #atomic units constants
        self.a0 = 5.2917721e-11
        m_e = 0.510e6/const.c**2
        self.Ry_eV = const.hbar**2/(const.e**2*m_e*self.a0**2)
        #convert into atomic units
        Delta = Delta/self.Ry_eV
        d = d/self.a0
        k = k*self.a0
        x = x/self.a0
        #generate energy array
        self.E= np.linspace(-Erange/self.Ry_eV,Erange/self.Ry_eV,Epx) +gamma*1j
        self.Er = np.real(self.E)
        #green function compute
        E = self.E
        m_e = 1
        kp = np.sqrt(k**2 + 2*m_e*E)
        km = np.sqrt(k**2  -2*m_e*E)
        FE = np.abs(E)/(np.sqrt(E**2-Delta**2))
        G =(m_e/(k))*((1j* FE* np.cos(kp*x+km*d) - np.sin(kp*x+km*d) )*( np.cos(kp*(x+d) ))/((1j*FE* np.sin((kp-km)*d)-np.cos((kp-km)*d)) ))
        return np.abs(np.imag(G))


