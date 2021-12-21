import numpy as np
import Green_functions as GF
import scipy as scy
import matplotlib.pyplot as plt
import time

gr=GF.Green()


def F(x,y,En,L,alpha,x_atom,y_atom):
    J1=L*0.2
    J2=L*0.2
    mode=1
    delta=1
    m=1
    pf=1
    gr.G(x,y,np.complex(En,0.001),delta,m,pf,mode)
    gr.dG(-x,-y,np.complex(En,0.001),x_atom,y_atom,J1,J2,alpha,delta,m,pf,mode)
    a=np.dot(gr.G0,gr.deltaG)
    return np.imag(np.trace(a)/L)

theta=np.linspace(0,np.pi,100)


mode=1
delta=1
m=1
pf=1

x=np.linspace(-1,1,1000)
y=np.linspace(-1,1,200)
omega=np.linspace(-2,2,1000)
L=np.linspace(0.00001,1,100)

I=0
start = time.time()

for i in omega:
    J1=0.5*0.2
    J2=0.5*0.2
    gr.G(1,1,np.complex(i,0.001),delta,m,pf,mode)
    gr.dG(-1,-1,np.complex(i,0.001),0,2,J1,J2,0,delta,m,pf,mode)
    a=gr.G0
    b=gr.deltaG


end = time.time()
print(end - start)

