import numpy as np
import scipy.special as scisp
import scipy.integrate as integ
import mpmath
from scipy import constants as const
import Green_functions as gr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
sim = gr.simulation()
import scipy as sc
import multiprocessing as mp
from detect_peaks import detect_peaks
from functools import partial


kf=0.7
hbar=1
m=1


def KN(w,s):
    return kf+s*(m/(hbar**2*kf))*np.sqrt(w**2)

def KS(w,delta,s):
    return kf+s*1j*(m/(hbar**2*kf))*np.sqrt(delta**2-w**2)

def V(J):
    return np.array([[J,0],[0,J]])

def GN(w,a,x,y):
    if x>=y:
        temp=lambda s: (m/hbar**2)*np.array([[(w+s*np.sqrt(w**2))/np.sqrt(w**2),0],[0,(w-s*np.sqrt(w**2))/np.sqrt(w**2)]])*(s*np.cos(KN(w,s)*x)*np.cos(KN(w,s)*(y+a)))/(np.sin(KN(w,s)*a)*KN(w,s))
        return temp(1)+temp(-1)
    else:
        temp=lambda s: (m/hbar**2)*np.array([[(w+s*np.sqrt(w**2))/np.sqrt(w**2),0],[0,(w-s*np.sqrt(w**2))/np.sqrt(w**2)]])*(s*np.cos(KN(w,s)*y)*np.cos(KN(w,s)*(x+a)))/(np.sin(KN(w,s)*a)*KN(w,s))
        return temp(1)+temp(-1)
#def GN(w,a,x,y):
#    temp=lambda s: ((m/hbar**2)*np.multiply([[(w+s*np.sqrt(w**2))/np.sqrt(w**2),0*w],[0*w,(w-s*np.sqrt(w**2))/np.sqrt(w**2)]],\
#       (s*np.cos(KN(w,s)*x)*np.cos(KN(w,s)*(y+a)))/(np.sin(KN(w,s)*a)*KN(w,s))))*np.heaviside(x-y,1)+\
#         ((m/hbar**2)*np.multiply([[(w+s*np.sqrt(w**2))/np.sqrt(w**2),0*w],[0*w,(w-s*np.sqrt(w**2))/np.sqrt(w**2)]],
#        (s*np.cos(KN(w,s)*y)*np.cos(KN(w,s)*(x+a)))/(np.sin(KN(w,s)*a)*KN(w,s))))*np.heaviside(-x+y,0)
#    return temp(1)+temp(-1)

def GS(w,delta,a,x,y):
    if x<=y:
        temp=lambda s: (m/hbar**2)*np.array([[(w+s*1j*np.sqrt(-w**2+delta**2)),delta],[delta,(w-s*1j*np.sqrt(-w**2+delta**2))]])*(-np.cos(KS(w,delta,s)*x)*np.exp(s*1j*KS(w,delta,s)*y))/(KS(w,delta,s)*np.sqrt(-w**2+delta**2))
        return temp(1)+temp(-1)
    else:
        temp=lambda s: (m/hbar**2)*np.array([[(w+s*1j*np.sqrt(-w**2+delta**2)),delta],[delta,(w-s*1j*np.sqrt(-w**2+delta**2))]])*(-np.cos(KS(w,delta,s)*y)*np.exp(s*1j*KS(w,delta,s)*x))/(KS(w,delta,s)*np.sqrt(-w**2+delta**2))
        return temp(1)+temp(-1)
#def GS(w,delta,a,x,y):
#    temp=lambda s: (m/hbar**2)*np.multiply([[(w+s*np.sqrt(w**2-delta**2))/np.sqrt(w**2-delta**2),delta/np.sqrt(w**2-delta**2)],[delta/np.sqrt(w**2-delta**2),(w-s*np.sqrt(w**2-delta**2))/np.sqrt(w**2-delta**2)]],\
#        (-1j*np.cos(KS(w,delta,s)*x)*np.exp(s*1j*KS(w,delta,s)*y))/(KS(w,delta,s)))*np.heaviside(x-y,1)+\
#            (m/hbar**2)*np.multiply([[(w+s*np.sqrt(w**2-delta**2))/np.sqrt(w**2-delta**2),delta/np.sqrt(w**2-delta**2)],[delta/np.sqrt(w**2-delta**2),(w-s*np.sqrt(w**2-delta**2))/np.sqrt(w**2-delta**2)]],\
#                (-1j*np.cos(KS(w,delta,s)*y)*np.exp(s*1j*KS(w,delta,s)*x))/(KS(w,delta,s)))*np.heaviside(-x+y,0)
#    return temp(1)+temp(-1)


def GTN(w,delta,a,x,y):
    return GN(w,a,x,y)-np.dot(np.dot(GN(w,a,x,0),np.linalg.inv(GS(w,delta,a,0,0)+GN(w,a,0,0))),GN(w,a,0,y))

def GTS(w,delta,a,x,y):
    return GS(w,delta,a,x,y)-np.dot(np.dot(GS(w,delta,a,x,0),np.linalg.inv(GS(w,delta,a,0,0)+GN(w,a,0,0))),GS(w,delta,a,0,y))

def GTSN(w,delta,a,x,y):
    return np.dot(np.dot(GS(w,delta,a,x,0),np.linalg.inv(GS(w,delta,a,0,0)+GN(w,a,0,0))),GN(w,a,0,y))

def GTNS(w,delta,a,x,y):
    return np.dot(np.dot(GN(w,a,x,0),np.linalg.inv(GS(w,delta,a,0,0)+GN(w,a,0,0))),GS(w,delta,a,0,y))

def G(w,delta,a,x,y):
    if x>=0 and y>=0:
        return GTS(w,delta,a,x,y)
    if x<0 and y<0:
        return GTN(w,delta,a,x,y)
    if x>=0 and y<0:
        return GTSN(w,delta,a,x,y)
    else:
        return GTNS(w,delta,a,x,y)

def GYSR(w,delta,J,a,x,y):
    return G(w,delta,a,x,y)+np.dot(np.dot(np.dot(G(w,delta,a,x,-a),V(J)),np.linalg.inv(np.identity(2)-np.dot(V(J),G(w,delta,a,-a,-a)))),G(w,delta,a,-a,y))

def SJdG(w,delta,a):
    return w*np.sin(a*2*m*w/(hbar**2*kf))-(np.sqrt(delta**2-w**2))*np.cos(a*2*m*w/(hbar**2*kf))

def YSR(w,delta,J,a):
    return (np.identity(2)-np.dot(V(J),G(w,delta,a,-a,-a)))[0,0]


def energies(J,a):
    w1=np.linspace(0.000,0.0009,2000)
    y=[]
    yy=[]
    for i in w1:
        y.append(-np.imag(np.trace(G(i+0.000001*1j,0.001,a,-a,-a))))
    SJdG_energy=w1[detect_peaks(y)[0]]
    w2=np.linspace(-SJdG_energy+0.00005,SJdG_energy+0.00005,2000)
    for i in w2:
        yy.append(-np.imag(np.trace(GYSR(i+0.000001*1j,0.001,J,a,-a,-a))))
    YSR_energy=w2[detect_peaks(yy)[0]]
    return [SJdG_energy,YSR_energy]


def resSJdG2(x,y,SJdG_energy,a):
    return  -np.imag(G(SJdG_energy+0.00001*1j,0.001,a,x,y))*0.00001+np.real(G(SJdG_energy+0.00001*1j,0.001,a,x,y))*0.00001*1j

def resYSR2(x,y,YSR_energy,J,a):
    return -np.imag(GYSR(YSR_energy+0.00001*1j,0.001,J,a,x,y))*0.00001+np.real(GYSR(YSR_energy+0.00001*1j,0.001,J,a,x,y))*0.00001*1j

def overlap(J,a):
    SJdG_energy,YSR_energy=energies(J,a)
    print(SJdG_energy,YSR_energy)
    x=np.linspace(-a,2000,500)
    z=np.linspace(-a,2000,500)

    u=0
    v=0
    for i in x:
        #print(i)
        for j in z:
            t=resSJdG2(i,j,SJdG_energy,a)
            tt=resYSR2(i,j,YSR_energy,J,a)
            u+=np.abs(t[0,0]*tt[0,0])
            v+=np.abs(t[1,1]*tt[1,1])
            #norm+=np.abs(resSJdG2(i,j,SJdG_energy)**2)
    return np.sqrt(u*((2000+a)/500)**2)+np.sqrt(v*((2000+a)/500)**2)



def main(n,i,n_cores=10):
    J= np.linspace(0.0,-kf,n)
    with mp.Pool(n_cores) as p:
        p_map = p.map(partial(overlap,a=i),J)
        y1 = list(p_map)
    return J,y1

if __name__ == "__main__":
    a=np.linspace(250,1000,5)
    file=open('data.dat','w')
    for i in a:
        print(i)
        x,y= main(10,i)
        print(x,y)
        file.write(str(x))
        file.write(str(y))
        plt.plot(x,np.array(y)/np.max(y))
    #plt.plot(x,yy)
    file.close()
    plt.show()


