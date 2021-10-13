from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from superconductor import BCS_curve, dynes_curve, dynes_curve_diff
from distributions import fermiDirac, fermiDirac_diff
import numpy as np
import pandas as pd
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def sum_data(bias,conductance,x_min,x_max):    
    #a=data[data==x_min].index[0]
    #b=data[data==x_max].index[0]
    a=find_nearest(bias,x_min)
    b=find_nearest(bias,x_max)
    mean1=0.0
    mean2=0.0
    for i in range(0,b):
        mean1=mean1+conductance[i]
    for i in range(a,len(bias)):
        mean2=mean2+conductance[i]

    mean1=mean1/float(b)
    mean2=mean2/(float(len(bias)-a))

    return [mean1,mean2]

def extensions(bias,conductance,x_min,x_max,N):
    sp=bias[1]-bias[0]
    mean=sum_data(bias,conductance,x_min,x_max)
    a=[bias[0]-sp]
    b=[bias[len(bias)-1]+sp]
    c=[mean[0]]
    d=[mean[1]]
    for i in range(N):
        a.insert(0,a[0]-sp)
        b.append(b[len(b)-1]+sp)
        c.append(mean[0])
        d.append(mean[1])
    a=pd.Series(a)
    b=pd.Series(b)
    c=pd.Series(c)
    d=pd.Series(d)
    bias=pd.Series(bias)
    conductance=pd.Series(conductance)
    bias=a.append(bias)
    bias=bias.append(b)
    conductance=c.append(conductance)
    conductance=conductance.append(d)
    return [bias,conductance]

def dynesDeconvolute(bias,conductance, gap=1.30e-3, temperature=1.248, dynesParameter=40e-6, energyR=6E-3, spacing=50e-6,x_min=-3.0E-3,x_max=3.0E-3,N=1000, window=15,order=3,n=3000):
    bias_new=extensions(bias,conductance,x_min,x_max,N)[0]
    conductance_n=extensions(bias,conductance,x_min,x_max,N)[1]
    conductance_new=savgol_filter(conductance_n,window,order)
    x=np.linspace(min(bias_new),max(bias_new),n)
    f2 = interp1d(bias_new, conductance_new, kind='cubic') 
    #dec_bias = np.arange(bias_new[np.round(len(bias_new)/2)]-energyR, bias_new[np.round(len(bias_new)/2)]+energyR, spacing)
    dec_bias = np.arange(-energyR,energyR, spacing)
    
    M_E, M_eV = np.meshgrid(dec_bias,x)
    M_N = dynes_curve(M_E,gap,dynesParameter)
    M_NV = dynes_curve(M_E+M_eV,gap,dynesParameter)
    M_dNV = dynes_curve_diff(M_E+M_eV,gap,dynesParameter)

    M_f = fermiDirac(M_E, temperature)
    M_fV = fermiDirac(M_E+M_eV, temperature)
    M_dfV = fermiDirac_diff(M_E+M_eV, temperature)
    M_g = M_dNV*(M_f-M_fV) - M_NV*M_dfV
    M_g_pinv = np.linalg.pinv(M_g)
    dec_conductance = np.dot(M_g_pinv,f2(x))
    return dec_bias,dec_conductance

def dynesDeconvolute_nof(bias,conductance,gap=1.25e-3, temperature=1.7, dynesParameter=40E-6, energyR=6E-3, spacing = 30e-6,x_min=-2.0E-3,x_max=2.0E-3,N=100):
    bias_new=extensions(bias,conductance,x_min,x_max,N)[0]
    conductance_new=extensions(bias,conductance,x_min,x_max,N)[1]

    #deconvolutionBias = np.arange(bias_new[np.round(len(bias_new)/2)]-energyR, bias_new[np.round(len(bias_new)/2)]+energyR, spacing)
    dec_bias = np.arange(-energyR,energyR, spacing)
    
    M_E, M_eV = np.meshgrid(dec_bias,bias_new)
    #print(conductance_new)
    M_N = dynes_curve(M_E,gap,dynesParameter)
    M_NV = dynes_curve(M_E+M_eV,gap,dynesParameter)
    #M_dN = dynes_curve_diff(M_E,gap,dynesParameter)
    M_dNV = dynes_curve_diff(M_E+M_eV,gap,dynesParameter)

    M_f = fermiDirac(M_E, temperature)
    M_fV = fermiDirac(M_E+M_eV, temperature)
    #M_df = fermiDirac_diff(M_E, temperature)
    M_dfV = fermiDirac_diff(M_E+M_eV, temperature)
    M_g = M_dNV*(M_f-M_fV) - M_NV*M_dfV
    #g_conv = np.trapz((M_g*M_N),deconvolutionBias)
    M_g_pinv = np.linalg.pinv(M_g)
    dec_conductance = np.dot(M_g_pinv,conductance_new)
    #g_reconv = np.dot(M_g,deconvolutionConductance)
    return dec_bias,dec_conductance