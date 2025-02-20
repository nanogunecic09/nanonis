import numpy as np
from scipy import constants as const
## PoE Superconductivity

Eh = const.physical_constants['Hartree energy in eV']

def f(t,x,R,C,T):
    x=np.linspace(0.01,5,2000)
    dx=x[1]-x[0]
    ReZ=R/(1+(x*R*C)**2)
    a,b=np.meshgrid(x,t)
    c=a*b
    return  np.dot(np.cos(c)-1,2*(ReZ/x)/np.tanh(x/(2*T)))*dx- np.dot(1j*np.sin(c) ,2*(ReZ/x) )*dx


def PD(R,C1,C2,T,n,N,Vmax):
    dis=(N/2-1)/(16*np.pi*Vmax*Eh)

    C=C1*C2/(C1+C2)
    if n==1:
        k=C2/(C1+C2)
    else:
        k=C1/(C1+C2)
    t=np.linspace(-dis,dis,N)
    x=np.linspace(0.01,10,100)
    freq=np.fft.fftfreq(N,d=2*dis/N)
    a=np.argsort(freq)
    y=np.real(np.fft.fft(np.exp(k*f(t,x,R,C,T))))
    t=np.abs(y[a])
    return np.flip(np.sort(freq)),t/sum(t)

def initD_(R1,R2,C1,C2,V,delta_t,Delta,T,eta,N):
    global EnD
    EnD=[0 for i in range(25)]
    global MnD
    MnD=[0 for i in range(25)]
    global x_int
    x_int=np.linspace(-4*np.max(V),4*np.max(V),N)
    a,b=np.meshgrid(x_int,x_int)
    t=a+b
    global bcsf
    bcsf=bcs(delta_t,t,eta)*fermi(T,t)

    global bcsif_even
    bcsif_even=bcs_i(Delta,t,eta,0)*fermi(T,t)

    global bcsif_odd
    bcsif_odd=bcs_i(Delta,t,eta,1)*fermi(T,t)

    f1,a1=PD(R1+R2,C1,C2,T*Eh,1,N,np.max(V))
    global freq1
    freq1=f1/(2*np.pi*27)
    global PoE1
    PoE1=a1
    f2,a2=PD(R1+R2,C1,C2,T*Eh,2,N,np.max(V))
    global freq2
    freq2=f2/(2*np.pi*27)
    global PoE2
    PoE2=a2

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
    return np.dot(bcs_i(Delta,t,eta,n)*(fermi(T,-t)),np.dot(bcsf,PoE1))/R1


def Gamma2D(V,R2,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta):
    a,b=np.meshgrid(x_int,E2(C1,C2,V,-n,-Q0))
    t=a+b
    if np.mod(n,2)==0:
        return np.dot(bcs_i(Delta,t,eta,n)*(fermi(T,-t)),np.dot(bcsif_even,PoE1))/R2
    else:
        return np.dot(bcs_i(Delta,t,eta,n)*(fermi(T,-t)),np.dot(bcsif_odd,PoE1))/R2


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

def all_pD(V,n):
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
    initD_(R1,R2,C1,C2,V,delta_t,Delta,T,eta,5000)
    n=check_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta)
    p0,pn,p_n=all_pD(V,n)
    I=p0*G1nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,0)
    for i in range(1,n+1):
        I+=pn[i-1]*G1nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,i)
        I+=p_n[i-1]*G1nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,-i)
    t=np.gradient(I)
    return t/np.sum(t)


def currentD2(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta):
    initD_(R1,R2,C1,C2,V,delta_t,Delta,T,eta,5000)
    n=check_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta)
    p0,pn,p_n=all_pD(V,n)
    I=p0*G2nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,0)
    for i in range(1,n+1):
        I+=pn[i-1]*G2nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,i)
        I+=p_n[i-1]*G2nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,-i)
    t=np.gradient(I)
    return t/np.sum(t)

