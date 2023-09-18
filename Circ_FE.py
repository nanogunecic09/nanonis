from gettext import find
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import constants as const
import scipy.special as scisp
import mpmath
from scipy.optimize import bisect
import math
import multiprocessing as mp

c = const.physical_constants['Hartree energy'][0]/const.e



def G0(r1,r2,E):
    s0 = np.array([[1,0],[0,1]])
    s1 = np.array([[0,1],[1,0]])
    s3 = np.array([[1,0],[0,-1]])
    delta=0.78e-3/c
    m=20.956
    pf=0.274
    w = np.sqrt(delta**2-E**2)
    x = np.sqrt((r1[0]-r2[0])**2+(r1[1]-r2[1])**2) 
    r=(pf+1j*w*m/pf)*x
    J0=scisp.jv(0,r) #bessel function 0th order u=argument
    H0=complex(mpmath.struveh(0,r)) #gives the struve function  
    BCS = E/w*np.kron(s0,s0) + delta/w*np.kron(s1,s0)
    return  (2*m/np.pi)*(BCS*np.real(J0+1j*H0)-1j*np.kron(s3,s0)*np.imag(J0 + 1j*H0))
    
def V(theta,J,U):
    s0 = np.array([[1,0],[0,1]])
    s1 = np.array([[0,1],[1,0]])
    s3 = np.array([[1,0],[0,-1]])
    return J * np.cos(theta)*np.kron(s0,s3) + J*np.sin(theta)*np.kron(s0,s1)+U*np.kron(s3,s0)
    
def M(E,theta_,r__,J_,U_,N):
    n=N
    M = np.zeros((4*n,4*n),dtype=complex)
    for i in range(0,4*n,4):
        for j in range(0,4*n,4):
            M[i:i+4,j:j+4] = np.dot(G0(r__[i//4],r__[j//4],E),V(theta_[j//4],J_[j//4],U_[j//4]))
    return M

def G0_(r_,E,r__,N): #r_ is the measurement point
    G0_ = np.zeros((4*N,4),dtype=complex)
    n=0
    for i in range(0,4*N,4):
        G0_[i:i+4,0:4] = G0(r__[n],r_,E)
        n+=1
    return G0_
    
def G(r_,E,theta_,r__,J_,U_,N):
    MM = np.linalg.inv(np.identity(N*4)-M(E,theta_,r__,J_,U_,N))
    GG = np.dot(MM,G0_(r_,E,r__,N))
    G = G0((0,0),(0,0),E)
    n=0
    for i in range(0,4*N,4):
        G += np.dot(np.dot( G0(r_,r__[n],E) , V(theta_[n] ,J_[n],U_[n])), GG[i:i+4,0:4] )
        n+=1
    return G

def DOS(r_,E,theta_,r__,J_,U_,N):
    return np.imag(G(r_,E,theta_,r__,J_,U_,N))

def f(w,j,theta_,r__,U_,N):
    J_=[j]*N
    rho1=DOS(r__[0],w+10e-7*1j,theta_,r__,J_,U_,N)
    rho2=DOS(r__[1],w+10e-7*1j,theta_,r__,J_,U_,N)
    return (rho1[0,0]-rho1[1,1])+np.cos(theta_[1])*(rho2[0,0]-rho2[1,1])+np.sin(theta_[1])*(rho2[0,1]+rho2[1,0]) 

"""
def det(E,theta_,r__,J_,U_,N):
    I=np.identity(4*N)
    return np.linalg.det(I-M(E,theta_,r__,J_,U_,N))

def roots(g,a,b,N):
    roots=[]
    x=np.linspace(a,b,N)
    for i in range(len(x)-1):
        f1=g(x[i])
        f2=g(x[i+1])
        if np.real(f1)*np.real(f2)>0:
            continue
        else:
            roots.append(bisect(g,x[i],x[i+1]))
    return roots

def rho(j,theta_,r__,U_,N):
    J_=[j]*N
    g=lambda x: det(x+10e-8*1j,theta_,r__,J_,U_,N)
    temp=roots(g, 0, delta,20)
    rho=[]
    for i in range(len(temp)-1):
        rho1=DOS(r__[0],temp[i]+20e-7*1j,theta_,r__,J_,U_,N)
        rho2=DOS(r__[1],temp[i]+20e-7*1j,theta_,r__,J_,U_,N)
        rho.append((rho1[0,0]-rho1[1,1])+np.cos(theta_[1])*(rho2[0,0]-rho2[1,1])+np.sin(theta_[1])*(rho2[0,1]+rho2[1,0]))
    return rho
"""

def double_Integral(xmin, xmax, ymin, ymax, nx, ny, A):

    dS = ((xmax-xmin)/(nx-1)) * ((ymax-ymin)/(ny-1))

    A_Internal = A[1:-1, 1:-1]

    # sides: up, down, left, right
    (A_u, A_d, A_l, A_r) = (A[0, 1:-1], A[-1, 1:-1], A[1:-1, 0], A[1:-1, -1])

    # corners
    (A_ul, A_ur, A_dl, A_dr) = (A[0, 0], A[0, -1], A[-1, 0], A[-1, -1])

    return dS * (np.sum(A_Internal)+ 0.5 * (np.sum(A_u) + np.sum(A_d) + np.sum(A_l) + np.sum(A_r))+ 0.25 * (A_ul + A_ur + A_dl + A_dr))



def _f2(k):
    delta=0.00078/c
    x=np.linspace(0,2*delta,50)
    xx=np.linspace(0,0.1,50)
    y=[]
    print('a')
    for i in x:
        yy=[]
        for j in xx:
            yy.append(f(i,j,[0,k],[(0.0,0.0),(6.3,0.0)],[0.0,0.0],2))
        y.append(yy)
    return double_Integral(0,2*delta,0.0,0.1,50,50,np.array(y))

def _f(k):
    delta=0.00078/c
    return integrate.romberg(lambda w: integrate.quad(lambda j: f(w,j,[0,k],[(0.0,0.0),(6.3,0.0)],[0.0,0.0],2),0,0.03)[0],0,2*delta)

def main(n,n_cores=10):
    theta= np.linspace(0,np.pi,n)

    #delta=0.00078/c

    #e=np.linspace(-delta,delta,100)
    y1=[]
    y2=[]
    a=[]
    #with mp.Pool(n_cores) as p:
    #    p_map = p.map(_f,theta)
    #    y2 = list(p_map)

    with mp.Pool(n_cores) as p:
        p_map = p.map(_f2,theta)
        y1 = list(p_map)
    

    return theta,y1

if __name__ == "__main__":
    x,y= main(10)
    
    plt.plot(x,y)
    #plt.plot(x,yy)
    plt.show()