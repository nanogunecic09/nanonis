# Superconductor analysis module
# nanoImaging group @ nanoGune

import numpy as np
from scipy import signal
# from lmfit import Model
import nanonis
import matplotlib.pyplot as plt
import scipy.constants as const
from .Green_functions import spin5_sq
from nanonis import deconvolution as deconv

# Evaluate the non-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array or value
#       * Delta is the superconducting gap at 0K


# def BCS_curve(energy, Delta):
#     Delta = np.complex(Delta)
#     density = np.sign(energy)*np.real(np.divide(energy, np.sqrt(np.power(energy, 2)-np.power(Delta, 2))))
#     return density

# Evaluate the Dynes-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array
#       * Delta is the superconducting gap at 0K
#       * dynesParameter is the Dynes factor


def dynes_curve(energy,Delta,dynesParameter):
    dynesParameter = complex(0, dynesParameter)
    density = np.sign(energy)*np.real(np.divide(energy-dynesParameter,np.sqrt(np.power(energy-dynesParameter, 2)-np.power(Delta, 2))))
    return density

def dynes_javi(energy,dynesParameter,Delta):
    dynesParameter = complex(0, dynesParameter)
    density = np.sign(energy)*np.real(np.divide(energy-dynesParameter,np.sqrt(np.power(energy-dynesParameter, 2)-np.power(Delta, 2))))
    return density

def dynesdos(E, Gamma, Delta): #dynes function
    dos = np.real((np.abs(E+1j*Gamma))/np.sqrt((E+1j*Gamma)**2-Delta**2))
    return np.abs(dos)

def dynes_wrong(E, Delta, Gamma): #dynes function
    dos = np.real((E+1j*Gamma)/np.sqrt((E+1j*Gamma)**2-Delta**2))
    return np.abs(dos)/np.abs(dos)[0]


def fdd( E, mu, T): #fermi Dirac function
    if T == 0:
        f = np.heaviside(-(E-mu), 1)
    else:
        f = 1/(1+np.exp((E-mu)/(const.k*T/const.e)))
    return f


# Evaluate the derivative Dynes-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array
#       * Delta is the superconducting gap at 0K
#       * dynesParameter is the Dynes factor


def dynes_curve_diff(energy, Delta, dynesParameter):
    dynesParameter = np.complex(0, dynesParameter)
    density_de = -np.sign(energy)*np.real(np.divide(np.square(Delta),np.power(np.square(energy-dynesParameter)-np.square(Delta), 1.5)))
    density_de = np.nan_to_num(density_de)
    return density_de



# Evaluate Tinkham model for Coulomb blockade

def Ip(Vp,R2,C1,C2,Q0,delta=0):
    n0 = np.round(((-C2*Vp+Q0-const.e/2)/const.e + (-C2*Vp+Q0+const.e/2)/const.e)/2)
    if (-const.e/2 + n0*const.e -Q0 )/C1-delta < Vp <  (const.e/2 + n0*const.e -Q0)/C1+delta:
        Ip = 0
    else:
        Ip = (1/(R2*(C1+C2)))*(-(n0*const.e-Q0)+C1*Vp-const.e*np.sign(Vp)/2)
    return Ip

def fdd( E, mu, T): #fermi Dirac function
    if T == 0:
        f = np.heaviside(-(E-mu), 1)
    else:
        f = 1/(1+np.exp((E-mu)/(const.k*T/const.e)))
    return f



def coulomb(R2,C1,C2,Q0,Vpx=300,T=1.3,Erange=4e-3):
    V = np.linspace(-Erange,Erange,Vpx)
    Vint = np.linspace(-Erange/2,Erange/2,Vpx)
    I = []
    for Vp in V:
        I.append(Ip(Vp,R2,C1,C2,Q0))
    It = []
    for i in Vint:
        It.append(np.trapz(np.gradient(I)*( fdd(V,i,T)-fdd(V,0,T) ),x=V))
    return np.gradient(It)




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
        self.conductance = conductance
        self.bias = bias

    def fdd(self, E, mu, T): #fermi Dirac function
        if T == 0:
            f = np.heaviside(-(E-mu), 1)
        else:
            f = 1/(1+np.exp((E-mu)/(const.k*T/const.e)))
        return f

    def dynesdos(self, E, Gamma, Delta): #dynes function
        dos = np.real((np.abs(E+1j*Gamma))/np.sqrt((E+1j*Gamma)**2-Delta**2))
        return np.abs(dos)


    def dynes_curve(self,energy,dynesParameter,Delta):
        dynesParameter = complex(0, dynesParameter)
        density = np.sign(energy)*np.real(np.divide(energy-dynesParameter,np.sqrt(np.power(energy-dynesParameter, 2)-np.power(Delta, 2))))
        return density

    def dynes_javi(self,energy,dynesParameter,Delta):
        dynesParameter = complex(0, dynesParameter)
        density = np.sign(energy)*np.real(np.divide(energy-dynesParameter,np.sqrt(np.power(energy-dynesParameter, 2)-np.power(Delta, 2))))
        return density

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



    def SIS(self,bias, T, Delta1, Gamma1, Delta2, Gamma2,A): #General calculation of the dos with dynes in the tip and sample, broadened by Fermi Dirac
        curr = []
        self.E = np.linspace(-Delta1*10,Delta1*10,5000)
        
        for Vp in self.bias:
            currp = np.trapz(self.dynesdos(self.E, Gamma1, Delta1)*self.dynesdos(self.E-Vp,
                             Gamma2, Delta2)*(self.fdd(self.E, Vp, T)-self.fdd(self.E, 0, T)), x=self.E)
            curr.append(currp)
        out = A*np.gradient(np.array(curr))
        return out/out[0]
    
    def SISjavi(self,bias, T, Delta1, Gamma1, Delta2, Gamma2,A): #General calculation of the dos with dynes in the tip and sample, broadened by Fermi Dirac
        curr = []
        self.E = np.linspace(-Delta1*10,Delta1*10,5000)
        
        for Vp in self.bias:
            currp = np.trapz(self.dynes_javi(self.E, Gamma1, Delta1)*self.dynes_javi(self.E-Vp,
                             Gamma2, Delta2)*(self.fdd(self.E, Vp, T)-self.fdd(self.E, 0, T)), x=self.E)
            curr.append(currp)
        out = A*np.gradient(np.array(curr))
        return out/out[0]

    def dynesconv(self,bias,T,Delta,Gamma):
        curr = []
        for Vp in self.biasHR:
            currp = np.trapz(self.dynesdos(self.biasHR, Gamma, Delta)*(self.conductance/self.conductance[0])*(self.fdd(self.biasHR, Vp, T)-self.fdd(self.biasHR, 0, T)),x=self.biasHR)
            curr.append(currp)
        return np.gradient(np.array(curr))

    def dynes_arnold_conv(self,x,T,Delta_s,Delta_t,gamma_t,gamma_s=1,d=1,k=1.2e10):
        curr = []
        bias = np.linspace(-4*(Delta_t), 4*(Delta_t), 1000)
        E = np.linspace(bias[0],bias[-1],1000)
        for Vp in bias:
            currp = np.trapz(self.dynesdos(E, gamma_t, Delta_t)*(self.arnolddos(E-Vp,gamma=gamma_s,d=d,k=k,Delta=Delta_s))*(self.fdd(E, Vp, T)-self.fdd(E, 0, T)),x=E)
            curr.append(currp)
        out = -np.gradient(np.array(curr))
        return bias, out/out[0]


    def YSRconv(self,bias,J,delta_s,delta_t,gamma_s,gamma_t): #spectra of a single ysr convoluted with dynes
        m=20.956
        pf=0.274
        c = const.physical_constants['Hartree energy'][0]/const.e
        
        #simulate YSR
        sim = spin5_sq(2 , [J,J] , [0,0] , ((0,0),(12,6,6.3)) ,m=m,pf=pf,delta=delta_s/c)
        conductance = []
        E = np.linspace(-8*delta_s ,8*delta_s,900)

        for i in E/c:
            conductance.append(np.sign(i)*sim.DOS((0,0),i+gamma_s*1j*np.sign(i)/c))
        conductance = np.array(conductance)
        #convolution with tip
        bias = np.linspace(-4*delta_s,4*delta_s,300)
        conv_cond = deconv.dynesConvolute(bias,E,conductance,delta_t,1.1,gamma_t)
        return conv_cond/conv_cond[0],bias

    def fitYSR(self):
        model = Model(self.YSRconv)
        params = model.make_params()
        params['J'].set(-0.0296,vary=False)
        params['delta_s'].set(0.71e-3,vary=False)
        params['delta_t'].set(1,vary=False,expr='delta_s')
        params['gamma_s'].set(60e-6,vary=False)
        params['gamma_t'].set(60e-6,vary=False)
        self.fit_res = model.fit(self.conductance,params,bias=self.bias)
        self.fit_res_eval = self.fit_res.eval(x=self.bias)


    def cond_spline():
        pass

    def fitSIS(self, T, Delta1, Gamma1, Delta2, Gamma2,A):
        model = Model(self.SIS)
        params = model.make_params()
        params['T'].set(T,vary=False)
        params['Delta1'].set(Delta1,vary=True)
        params['Gamma1'].set(Gamma1,vary=True)
        params['Delta2'].set(Delta2,vary=False)
        params['Gamma2'].set(Gamma2,vary=False)
        params['A'].set(A,vary=False)

        self.fit_res = model.fit(self.conductance,params,bias=self.bias)
        self.fit_res_eval = self.fit_res.eval(x=self.bias)
        return 
    
    def fit_arnold(self):
        model = Model(self.dynes_arnold_conv)
        self.params = model.make_params()
        self.params['T'].set(1,vary=False)
        self.params['Delta'].set(0.75e-3,vary=True)
        self.params['Gamma'].set(1e-6,min=0,vary=False)
        self.params['gamma'].set(4e-6,min=0,vary=True)
        self.params['d'].set(8e-5,vary=True)
        self.params['k'].set(3.2e12,vary=False)
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

