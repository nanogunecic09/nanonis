import numpy as np
from scipy import constants as const


Eh = const.physical_constants['Hartree energy in eV']
## Improoved No PoF Superconductivity
def initD_(V,delta_t,Delta,T,eta,eta_t):
    global EnD
    EnD=[0 for i in range(25)]
    global MnD
    MnD=[0 for i in range(25)]
    global x_int
    x_int=np.linspace(-4*np.max(V),4*np.max(V),5000)

    global bcsf
    bcsf=bcs(delta_t,x_int,eta_t)*fermi(T,x_int)

    global bcsif_even
    bcsif_even=bcs_i(Delta,x_int,eta,0)*fermi(T,x_int)

    global bcsif_odd
    bcsif_odd=bcs_i(Delta,x_int,eta,1)*fermi(T,x_int)

def bcs(delta,x,eta):
    if delta<=0.01: 
        return np.full(np.shape(x),1)
    else:
        return (np.sign(x))*np.imag(np.divide(np.abs(x+eta*1j),np.sqrt(delta**2-(x+eta*1j)**2)))
    
def bcs_i(delta,x,eta,n):
    if np.mod(n,2)==0:
        if delta<=0.01: 
            return np.full(np.shape(x),1)
        else:
            return (np.sign(x))*np.imag(np.divide(np.abs(x+eta*1j),np.sqrt(delta**2-(x+eta*1j)**2)))
    else:
        return 1
        

def fermi(T,x):
    if T==0.0:
        return np.heaviside(-x,1)
    else:
        return np.divide(1,1+np.exp(x/T))

def E0(C1,C2,V,n,Q0,Delta):
    return (n+Q0)**2/(2*(C1+C2))+(1-(-1)**(n))*Delta/2

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

def Gamma2D(V,R2,C1,C2,n,Q0,delta_s,delta_t,T,eta):
    a,b=np.meshgrid(x_int,E2(C1,C2,V,-n,-Q0))
    t=a+b
    if np.mod(n,2)==0:
        return np.dot(bcs(delta_s,t,eta)*(fermi(T,-t)),bcsif_even)/R2
    else:
        return np.dot(bcs(delta_s,t,eta)*(fermi(T,-t)),bcsif_odd)/R2



def PND(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n):
    mn=(Gamma1D(V,R1,C1,C2,-n,Q0,Delta,delta_t,delta_s,T,eta)+Gamma2D(-V,R2,C1,C2,n,-Q0,Delta,delta_t,delta_s,T,eta))/(Gamma1D(-V,R1,C1,C2,n+1,-Q0,Delta,delta_t,delta_s,T,eta)+Gamma2D(V,R2,C1,C2,-n-1,Q0,Delta,delta_t,delta_s,T,eta))
    en=(Gamma1D(-V,R1,C1,C2,-n,-Q0,Delta,delta_t,delta_s,T,eta)+Gamma2D(V,R2,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta))/(Gamma1D(V,R1,C1,C2,n+1,Q0,Delta,delta_t,delta_s,T,eta)+Gamma2D(-V,R2,C1,C2,-n-1,-Q0,Delta,delta_t,delta_s,T,eta))
    return en,mn

def check_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta):
    V=V[-1]
    n=[0]
    while True:
        Z=0
        for i in range(-n[-1],n[-1]+1):
            Z+=np.exp(-E0(C1,C2,V,i,Q0,Delta)/T)
        pn=np.exp(-E0(C1,C2,V,n[-1],Q0,Delta)/T)/Z
        p_n=np.exp(-E0(C1,C2,V,-n[-1],Q0,Delta)/T)/Z
        n.append(n[-1]+1)
        if pn<0.01 and p_n<0.01:
            #print(len(a))
            break 
    return n[-1]

def all_pD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n):
    Z=0
    for i in range(-n,n+1):
        Z+=np.exp(-E0(C1,C2,V,i,Q0,Delta)/T)
    p0=np.exp(-E0(C1,C2,V,0,Q0,Delta)/T)/Z
    pn=[]
    p_n=[]
    for i in range(1,n+1):
        pn.append(np.exp(-E0(C1,C2,V,i,Q0,Delta)/T)/Z)
        p_n.append(np.exp(-E0(C1,C2,V,-i,Q0,Delta)/T)/Z)
    return p0,pn,p_n

def G1nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n):
    return Gamma1D(V,R1,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta)-Gamma1D(-V,R1,C1,C2,-n,-Q0,Delta,delta_t,delta_s,T,eta)

def G2nD(V,R1,R2,C1,C2,Q0,Delta,delta_t,delta_s,T,eta,n):
    return Gamma2D(V,R2,C1,C2,n,Q0,Delta,delta_t,delta_s,T,eta)-Gamma2D(-V,R2,C1,C2,-n,-Q0,Delta,delta_t,delta_s,T,eta)

def currentD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,eta_t):
    initD_(V,Delta_t,Delta,T,eta,eta_t)
    n=check_pD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta)
    p0,pn,p_n=all_pD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,n)
    I=p0*G1nD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,0)
    for i in range(1,n+1):
        I+=pn[i-1]*G1nD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,i)
        I+=p_n[i-1]*G1nD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,-i)
    t=np.gradient(I)
    return t/t[0]


def currentD2(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,eta_t):
    initD_(V,Delta_t,Delta,T,eta,eta_t)
    n=check_pD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta)
    p0,pn,p_n=all_pD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,n)
    I=p0*G2nD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,0)
    for i in range(1,n+1):
        I+=pn[i-1]*G2nD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,i)
        I+=p_n[i-1]*G2nD(V,R1,R2,C1,C2,Q0,Delta,Delta_t,Delta_s,T,eta,-i)
    t=np.gradient(I)
    return t/t[0]


import matplotlib.pyplot as plt
import nanonis

spectra = nanonis.biasSpectroscopy()

def fitter(fname,initial_params=None,off=0):
    fnames = fname
    if initial_params == None:
            p = {'R1': 1000,
        'R2': 1,
        'C1': 0.002,
        'C2': 3,
        'T': 0.07,
        'Q0': -0.75,
        'Delta': 1.35,
        'Delta_t': 1.35,
        'Delta_s': 0.3,
        'eta': 0.08,
        'eta_t': 0.08}
    else:
        p = initial_params


    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,10)))

    last=True




    fig,ax=plt.subplots()
    fig.subplots_adjust(bottom=0.7)

    def plotdata(fnames):
        for f in fnames:
            spectra.load(f)
            spectra.biasOffset(off*1e-3)
            spectra.normalizeRange_symm([4e-3,5e-3])
            b1, c1 = spectra.bias*1e3,spectra.conductance
            ax.plot(b1, c1)

    plotdata(fnames)

    # current calc
    x=np.linspace(-5,5,100)
    y=currentD(x,300,1,0.01,0.2,0.0,1.3,1.3,0.1,0.1,0.05,0.05)
    ax.plot(x,y/y[0])

    ax1 = fig.add_axes([0.10, 0.10, 0.65, 0.03])
    ax2 = fig.add_axes([0.10, 0.15, 0.65, 0.03])
    ax3 = fig.add_axes([0.10, 0.20, 0.65, 0.03])
    ax4 = fig.add_axes([0.10, 0.25, 0.65, 0.03])
    ax5 = fig.add_axes([0.10, 0.30, 0.65, 0.03])
    ax6 = fig.add_axes([0.10, 0.35, 0.65, 0.03])
    ax7 = fig.add_axes([0.10, 0.40, 0.65, 0.03])
    ax8 = fig.add_axes([0.10, 0.45, 0.65, 0.03])
    ax9 = fig.add_axes([0.10, 0.5, 0.65, 0.03])
    ax10 = fig.add_axes([0.10,0.55, 0.65, 0.03])
    ax11 = fig.add_axes([0.10,0.60, 0.65, 0.03])


    if last==False:
        ax1_s = plt.Slider(ax1,'R1',1,1000,valinit=300)
        ax2_s = plt.Slider(ax2,'R2',1,1000,valinit=1)
        ax3_s = plt.Slider(ax3,'C1',0.00001,0.2,valinit=0.01)
        ax4_s = plt.Slider(ax4,'C2',0.001,5,valinit=0.1)
        ax5_s = plt.Slider(ax5,'T',0,1,valinit=0.1)
        ax6_s = plt.Slider(ax6,'Q0',-1,1,valinit=0)
        ax7_s = plt.Slider(ax7,'Delta',0,1.3,valinit=1.3)
        ax8_s = plt.Slider(ax8,'Delta_t',0,1.5,valinit=1.3)
        ax9_s = plt.Slider(ax9,'Delta_s',0,1.3,valinit=0.3)
        ax10_s = plt.Slider(ax10,'eta',0.01,0.2,valinit=0.05)
        ax11_s = plt.Slider(ax11,'eta_t',0.01,0.2,valinit=0.05)



    if last==True:
        params = p
        # params = {
        # 'R1' : ax1_s.val,
        # 'R2' : ax2_s.val,
        # 'C1' : ax3_s.val,
        # 'C2' : ax4_s.val,
        # 'T' : ax5_s.val,
        # 'Q0' : ax6_s.val,
        # 'Delta' : ax7_s.val,
        # 'Delta_t' : ax8_s.val,
        # 'Delta_s' : ax9_s.val,
        # 'eta' : ax10_s.val
        # }
        ax1_s = plt.Slider(ax1,'R1',1,1000,valinit=params['R1'])
        ax2_s = plt.Slider(ax2,'R2',1,1000,valinit=params['R2'])
        ax3_s = plt.Slider(ax3,'C1',0.000001,0.2,valinit=params['C1'])
        ax4_s = plt.Slider(ax4,'C2',0.001,5,valinit=params['C2'])
        ax5_s = plt.Slider(ax5,'T',0,1,valinit=params['T'])
        ax6_s = plt.Slider(ax6,'Q0',-2,2,valinit=params['Q0'])
        ax7_s = plt.Slider(ax7,'Delta',0,1.5,valinit=params['Delta'])
        ax8_s = plt.Slider(ax8,'Delta_t',0,1.5,valinit=params['Delta_t'])
        ax9_s = plt.Slider(ax9,'Delta_s',0,1.5,valinit=params['Delta_s'])
        ax10_s = plt.Slider(ax10,'eta',0.01,0.2,valinit=params['eta'])
        ax11_s = plt.Slider(ax11,'eta_t',0.01,0.2,valinit=0.05)

    def update(val):
        ax.clear()
        y=currentD(x,ax1_s.val,ax2_s.val,ax3_s.val,ax4_s.val,ax6_s.val,ax7_s.val,ax8_s.val,ax9_s.val,ax5_s.val,ax10_s.val,ax11_s.val)
        ax.plot(x,y/y[0],color='k')
        plotdata(fnames)
        params = {
        'R1' : ax1_s.val,
        'R2' : ax2_s.val,
        'C1' : ax3_s.val,
        'C2' : ax4_s.val,
        'T' : ax5_s.val,
        'Q0' : ax6_s.val,
        'Delta' : ax7_s.val,
        'Delta_t' : ax8_s.val,
        'Delta_s' : ax9_s.val,
        'eta' : ax10_s.val
        }

    ax1_s.on_changed(update)
    ax2_s.on_changed(update)
    ax3_s.on_changed(update)
    ax4_s.on_changed(update)
    ax5_s.on_changed(update)
    ax6_s.on_changed(update)
    ax7_s.on_changed(update)
    ax8_s.on_changed(update)
    ax9_s.on_changed(update)
    ax10_s.on_changed(update)
    ax11_s.on_changed(update)

    return


def E_odd(V,n,C,Delta):
    return (n-V*C)**2/(2*C)+Delta

def E_even(V,n,C):
    return (n-V*C)**2/(2*C)
