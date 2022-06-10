import detect_peaks as dp
import nanonis
import distributions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib import rc
from scipy.signal import savgol_filter
import glob
import colorcet as cc
import pickle
import scipy.integrate as int
from scipy import constants as const
from lmfit.models import LorentzianModel, ConstantModel, GaussianModel

c = const.physical_constants['Hartree energy'][0]/const.e




def G0(r1,r2,E):
    N=2
    s0 = np.array([[1,0],[0,1]])
    s1 = np.array([[0,1],[1,0]])
    s3 = np.array([[1,0],[0,-1]])
    x1=r1[0]-r2[0]
    x2=r1[1]-r2[1]
    delta=0.78e-3/c
    m=20.956
    pf=0.274
    w = np.sqrt(delta**2-E**2)
    BCS = E/w*np.kron(s0,s0) + delta/w*np.kron(s1,s0)
    xi=np.kron(s3,s0)
    if np.abs(x1)<0.00001 and np.abs(x2)<0.00001:
            G1=(2*m/np.pi)*BCS
            G2=0.0
    elif np.abs(x2)<0.00001:
        G1=(np.exp(-(m*w/pf)*np.abs(x1))*( (1/np.abs(x1))*pf*np.sin(pf*(np.abs(x1)))+(pf**2)*np.cos(pf*np.abs(x1)) ) )*(m/(np.pi*pf**2))*BCS
        G2=(np.exp(-(m*w/pf)*np.abs(x1))*( (1/np.abs(x1))*pf*np.cos(pf*(np.abs(x1)))+(pf**2)*np.sin(pf*np.abs(x1)) ) -pf/np.abs(x1))*(m/(np.pi*pf**2))*xi
    elif np.abs(x1)<0.00001:
        G1=(np.exp(-(m*w/pf)*np.abs(x2))*( (1/np.abs(x2))*pf*np.sin(pf*(np.abs(x2)))+(pf**2)*np.cos(pf*np.abs(x2)) ) )*(m/(np.pi*pf**2))*BCS
        G2=(np.exp(-(m*w/pf)*np.abs(x2))*( (1/np.abs(x2))*pf*np.cos(pf*(np.abs(x2)))+(pf**2)*np.sin(pf*np.abs(x2)) ) -pf/np.abs(x2))*(m/(np.pi*pf**2))*xi
    elif x1+x2>=0.00001 and x1-x2>=0.00001:
        G1=(np.exp(-(m*w/pf)*(x1+x2))*(1/x1+1/x2)*pf*np.sin(pf*(x1+x2))+np.exp(-(m*w/pf)*(x1-x2))*(1/x1-1/x2)*pf*np.sin(pf*(x1-x2)) )*(m/(2*np.pi*pf**2))*BCS
        G2=(np.exp(-(m*w/pf)*(x1+x2))*(1/x1+1/x2)*pf*np.cos(pf*(x1+x2))+np.exp(-(m*w/pf)*(x1-x2))*(1/x1-1/x2)*pf*np.cos(pf*(x1-x2)) -2*pf/x1)*(m/(2*np.pi*pf**2))*xi
    elif x1+x2>0.00001 and x1-x2<0.00001:
        G1=(np.exp(-(m*w/pf)*(x1+x2))*(1/x1+1/x2)*pf*np.sin(pf*(x1+x2))+np.exp(-(m*w/pf)*(x2-x1))*(1/x2-1/x1)*pf*np.sin(pf*(x2-x1)) )*(m/(2*np.pi*pf**2))*BCS
        G2=(np.exp(-(m*w/pf)*(x1+x2))*(1/x1+1/x2)*pf*np.cos(pf*(x1+x2))+np.exp(-(m*w/pf)*(x2-x1))*(1/x2-1/x1)*pf*np.cos(pf*(x2-x1)) -2*pf/x2)*(m/(2*np.pi*pf**2))*xi
    elif x1+x2<=0.00001 and x1-x2<=0.00001:
        G1=(np.exp((m*w/pf)*(x1+x2))*(-1/x1-1/x2)*pf*np.sin(pf*(-x1-x2))+np.exp(-(m*w/pf)*(-x1+x2))*(-1/x1+1/x2)*pf*np.sin(pf*(-x1+x2)) )*(m/(2*np.pi*pf**2))*BCS
        G2=(np.exp((m*w/pf)*(x1+x2))*(-1/x1-1/x2)*pf*np.cos(pf*(-x1-x2))+np.exp(-(m*w/pf)*(-x1+x2))*(-1/x1+1/x2)*pf*np.cos(pf*(-x1+x2)) +2*pf/x1)*(m/(2*np.pi*pf**2))*xi
    elif x1+x2<0.00001 and x1-x2>0.00001:
        G1=(np.exp((m*w/pf)*(x1+x2))*(-1/x1-1/x2)*pf*np.sin(pf*(-x1-x2))+np.exp(-(m*w/pf)*(-x2+x1))*(-1/x2+1/x1)*pf*np.sin(pf*(-x2+x1)) )*(m/(2*np.pi*pf**2))*BCS
        G2=(np.exp((m*w/pf)*(x1+x2))*(-1/x1-1/x2)*pf*np.cos(pf*(-x1-x2))+np.exp(-(m*w/pf)*(-x2+x1))*(-1/x2+1/x1)*pf*np.cos(pf*(-x2+x1)) +2*pf/x2)*(m/(2*np.pi*pf**2))*xi
    return G1+G2
    
def V(theta,J,U):
    s0 = np.array([[1,0],[0,1]])
    s1 = np.array([[0,1],[1,0]])
    s3 = np.array([[1,0],[0,-1]])
    N=2
    return J * np.cos(theta)*np.kron(s0,s3) + J*np.sin(theta)*np.kron(s0,s1)+U*np.kron(s3,s0)
    
def M(E,theta_,r__,J_,U_):
    N=2
    n=N
    M = np.zeros((4*n,4*n),dtype=complex)
    for i in range(0,4*n,4):
        for j in range(0,4*n,4):
            M[i:i+4,j:j+4] = np.dot(G0(r__[i//4],r__[j//4],E),V(theta_[j//4],J_[j//4],U_[j//4]))
    return M

def G0_(r_,E,r__): #r_ is the measurement point
    N=2
    G0_ = np.zeros((4*N,4),dtype=complex)
    n=0
    for i in range(0,4*N,4):
        G0_[i:i+4,0:4] = G0(r__[n],r_,E)
        n+=1
    return G0_
    
def G(r_,E,theta_,r__,J_,U_):
    N=2
    MM = np.linalg.inv(np.identity(N*4)-M(E,theta_,r__,J_,U_))
    GG = np.dot(MM,G0_(r_,E,r__))
    G = G0((0,0),(0,0),E)
    n=0
    for i in range(0,4*N,4):
        G += np.dot(np.dot( G0(r_,r__[n],E) , V(theta_[n] ,J_[n],U_[n])), GG[i:i+4,0:4] )
        n+=1
    return G

def DOS(r_,E,theta_,r__,J_,U_):
    N=2
    return np.imag(G(r_,E,theta_,r__,J_,U_))

def f(j,w,theta_,x,y,r__,u):
    J_=[j,j]
    U_=[u,u]
    rho1=DOS((0,0),w+20e-7*1j,theta_,r__,J_,U_)
    rho2=DOS((x,y),w+20e-7*1j,theta_,r__,J_,U_)
    return j*(rho1[0,0]-rho1[1,1])+j*np.cos(theta_[1])*(rho2[0,0]-rho2[1,1])+j*np.sin(theta_[1])*(rho2[0,1]+rho2[1,0]) 


N = 2
theta= np.linspace(0,np.pi,10)
j=np.linspace(0,0.0025,10)
x=np.linspace(0,63.0,20)
delta=0.00078/c

e=np.linspace(-2*delta,2*delta,100)
y1=[]
y2=[]
for k in theta:
    print(k)
    #for i in theta:
    #print(k)
    y1.append(int.dblquad(f,0.0,0.0025,0.0,4*delta,args=([0,k],6.3,0.0,[(0,0),(6.3,0)],0.0),epsabs=1e-3)[0])
    #y2.append(int.dblquad(f,0.0,0.0025,0.0,4*delta,args=([0,np.pi],6.3,0.0,[(0,0),(k,0)],0.0))[0])

plt.plot(theta,y1)
#plt.plot(x,y2)
plt.show()