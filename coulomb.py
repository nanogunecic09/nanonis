import numpy as np
from scipy import constants as const


Eh = const.physical_constants['Hartree energy in eV']
## Improoved No PoF Superconductivity


def initD_(V,delta_t,Delta,T,eta):
    global EnD
    EnD=[0 for i in range(25)]
    global MnD
    MnD=[0 for i in range(25)]
    global x_int
    x_int=np.linspace(-4*np.max(V),4*np.max(V),5000)

    global bcsf
    bcsf=bcs(delta_t,x_int,eta)*fermi(T,x_int)

    global bcsif_even
    bcsif_even=bcs_i(Delta,x_int,eta,0)*fermi(T,x_int)

    global bcsif_odd
    bcsif_odd=bcs_i(Delta,x_int,eta,1)*fermi(T,x_int)

def bcs(delta,x,eta):
    if delta<=0.01: 
        return 1
    else:
        return (np.sign(x))*np.imag(np.divide(np.abs(x+eta*1j),np.sqrt(delta**2-(x+eta*1j)**2)))
    
def bcs_i(delta,x,eta,n):
    if np.mod(n,2)==0:
        if delta<=0.01: 
            return 1
        else:
            return (np.sign(x))*np.imag(np.divide(np.abs(x+eta*1j),np.sqrt(delta**2-(x+eta*1j)**2)))
    else:
        return 1
        

def fermi(T,x):
    if T==0.0:
        return np.heaviside(-x,1)
    else:
        return np.divide(1,1+np.exp(x/T))

def E1(C1,C2,V,n,Q0):
    k=C2/(C1+C2)
    return k*V+(n+Q0-1/2)/(C1+C2)

def E2(C1,C2,V,n,Q0):
    k=C1/(C1+C2)
    return k*V+(n+Q0-1/2)/(C1+C2)

def Gamma1D(V,R1,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta):
    a,b=np.meshgrid(x_int,E1(C1,C2,V,n,Q0))
    t=a+b
    return np.dot( bcs_i(Delta,t,eta,n)*(fermi(T,-t)),bcsf )/R1

def Gamma2D(V,R2,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta):
    a,b=np.meshgrid(x_int,E2(C1,C2,V,-n,-Q0))
    t=a+b
    if np.mod(n,2)==0:
        return np.dot(bcs(delta_s,t,eta)*(fermi(T,-t)),bcsif_even)/R2
    else:
        return np.dot(bcs(delta_s,t,eta)*(fermi(T,-t)),bcsif_odd)/R2


def PND(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n):
    mn=(Gamma1D(V,R1,C1,C2,-n,Q0,Delta,delta_t,delta_s,T,eta)+Gamma2D(-V,R2,C1,C2,n,-Q0,Delta,delta_t,delta_s,T,eta))/(Gamma1D(-V,R1,C1,C2,n+1,-Q0,Delta,delta_t,delta_s,T,eta)+Gamma2D(V,R2,C1,C2,-n-1,Q0,Delta,delta_t,delta_s,T,eta))
    en=(Gamma1D(-V,R1,C1,C2,-n,-Q0,Delta,delta_t,delta_s,T,eta)+Gamma2D(V,R2,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta))/(Gamma1D(V,R1,C1,C2,n+1,Q0,Delta,delta_t,delta_s,T,eta)+Gamma2D(-V,R2,C1,C2,-n-1,-Q0,Delta,delta_t,delta_s,T,eta))
    global En
    EnD[n]= en
    global Mn
    MnD[n]= mn
    return en,mn

def check_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta):
    n=[0]
    while True:
        a=[]
        b=[]
        for i in n:
            an,bn=PND(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,i)
            a.append(an[0])
            b.append(bn[0])
        p0=1
        for i in range(len(a)):
            temp=np.zeros(len(a))
            temp[0:i+1]=1
            temp=temp.tolist()
            p0+=np.prod(a,where=temp)+np.prod(b,where=temp)
        p0=1/p0
        pn=p0*np.prod(a)
        p_n=p0*np.prod(b)
        #print(pn,p_n)
        n.append(n[-1]+1)
        if pn<0.01 and p_n<0.01:
            #print(len(a))
            break 
    return n[-1]

def all_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n):
    a=[]
    b=[]
    for i in range(n+1):
        a.append(EnD[i])
        b.append(MnD[i])
    p0=1
    pn=[]
    p_n=[]
    for i in range(n):
        temp1=np.full(len(V),1.0)
        temp2=np.full(len(V),1.0)
        for j in range(i+1):
            temp1*=a[j]     
            temp2*=b[j]
        p0+=temp1+temp2
        pn.append(temp1)
        p_n.append(temp2)
    p0=1/p0
    pn=p0*np.array(pn)
    p_n=p0*np.array(p_n)
    return p0,pn,p_n

def G1nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n):
    return Gamma1D(V,R1,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta)-Gamma1D(-V,R1,C1,C2,-n,-Q0,Delta,delta_t,delta_s,T,eta)

def G2nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n):
    return Gamma2D(V,R2,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta)-Gamma2D(-V,R2,C1,C2,-n,-Q0,Delta,delta_t,delta_s,T,eta)

def currentD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta):
    initD_(V,delta_t,Delta,T,eta)
    n=check_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta)
    p0,pn,p_n=all_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n)
    I=p0*G1nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,0)
    for i in range(1,n+1):
        I+=pn[i-1]*G1nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,i)
        I+=p_n[i-1]*G1nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,-i)
    t=np.gradient(I)
    return t/np.sum(t)


def currentD2(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta):
    initD_(V,delta_t,Delta,T,eta)
    n=check_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta)
    p0,pn,p_n=all_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n)
    I=p0*G2nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,0)
    for i in range(1,n+1):
        I+=pn[i-1]*G2nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,i)
        I+=p_n[i-1]*G2nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,-i)
    t=np.gradient(I)
    return t/np.sum(t)
