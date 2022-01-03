import numpy as np
import matplotlib.pyplot as plt
import nanonis
from matplotlib.gridspec import GridSpec

def M(Del,D,J,E,p):
    Mat1=[[(25/4)*D + (5/4)*J, 0, np.sqrt(10)*E, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, (9/4)*D + (3/4)*J, 0, 3*np.sqrt(2)*E, 0, 0, p*(J/2)*np.sqrt(5), 0, 0, 0, 0, 0],
    [np.sqrt(10)*E, 0, (1/4)*D + (1/4)*J, 0, 3*np.sqrt(2)*E, 0, 0, p*J*np.sqrt(2), 0, 0, 0, 0], 
    [0, 3*np.sqrt(2)*E, 0, (1/4)*D - (1/4)*J, 0, np.sqrt(10)*E, 0, 0, p*(J/2)*3, 0, 0, 0], 
    [0, 0, 3*np.sqrt(2)*E, 0, (9/4)*D - (3/4)*J, 0, 0, 0, 0, p*J*np.sqrt(2), 0, 0], 
    [0, 0, 0, np.sqrt(10)*E, 0, (25/4)*D - (5/4)*J, 0, 0, 0, 0, p*(J/2)*np.sqrt(5), 0], 
    [0, p*(J/2)*np.sqrt(5), 0, 0, 0, 0, (25/4)*D - (5/4)*J, 0, np.sqrt(10)*E, 0, 0, 0], 
    [0, 0, p*J*np.sqrt(2), 0, 0, 0, 0, (9/4)*D - (3/4)*J, 0, 3*np.sqrt(2)*E, 0, 0], 
    [0, 0, 0, p*(3*J/2), 0, 0, np.sqrt(10)*E, 0, (1/4)*D - (1/4)*J, 0, 3*np.sqrt(2)*E, 0], 
    [0, 0, 0, 0, p*J*np.sqrt(2), 0, 0, 3*np.sqrt(2)*E, 0, (1/4)*D + (1/4)*J, 0, np.sqrt(10)*E], 
    [0, 0, 0, 0, 0, p*(J/2)*np.sqrt(5), 0, 0, 3*np.sqrt(2)*E, 0, (9/4)*D + (3/4)*J, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(10)*E, 0, (25/4)*D + (5/4)*J]]
    
    Mat0=[[(25/4)*D, 0, np.sqrt(10)*E, 0, 0, 0, Del, 0, 0, 0, 0, 0], 
    [0, (9/4)*D, 0, 3*np.sqrt(2)*E, 0, 0, 0, Del, 0, 0, 0, 0], 
    [np.sqrt(10)*E, 0, (1/4)*D, 0, 3*np.sqrt(2)*E, 0, 0, 0, Del, 0, 0, 0], 
    [0, 3*np.sqrt(2)*E, 0, (1/4)*D, 0, np.sqrt(10)*E, 0, 0, 0, Del, 0, 0], 
    [0, 0, 3*np.sqrt(2)*E, 0, (9/4)*D, 0, 0, 0, 0, 0, Del, 0], 
    [0, 0, 0, np.sqrt(10)*E, 0, (25/4)*D, 0, 0, 0, 0, 0, Del], 
    [Del, 0, 0, 0, 0, 0, (25/4)*D, 0, np.sqrt(10)*E, 0, 0, 0], 
    [0, Del, 0, 0, 0, 0, 0, (9/4)*D, 0, 3*np.sqrt(2)*E, 0, 0], 
    [0, 0, Del, 0, 0, 0, np.sqrt(10)*E, 0, (1/4)*D, 0, 3*np.sqrt(2)*E, 0], 
    [0, 0, 0, Del, 0, 0, 0, 3*np.sqrt(2)*E, 0, (1/4)*D, 0, np.sqrt(10)*E], 
    [0, 0, 0, 0, Del, 0, 0, 0, 3*np.sqrt(2)*E, 0, (9/4)*D, 0], 
    [0, 0, 0, 0, 0, Del, 0, 0, 0, np.sqrt(10)*E, 0, (25/4)*D]]
    return [Mat0,Mat1]

    
def peak(En,Gamma,DeltaE,w):
    return w*np.divide(Gamma,np.add(np.power(np.add(En,-DeltaE),2),Gamma**2/2))

def thermal(E1,E2,E):
    beta=20
    a=np.exp(-E1*beta)+np.exp(-E2*beta)
    return np.exp(-E*beta)/a

cpup=np.kron([[0,1],[0,0]],np.identity(6))
cpdown01=np.kron([[0,0],[0,1]],np.identity(6))
cpdown10=np.kron([[1,0],[0,0]],np.identity(6))

#Material parameters
Del=1
Deltat = 0.68

dpx = 100
epx = 100
jpx = 100

totpx = dpx + epx + jpx*3

D = np.linspace(0,0.9,dpx)
E = np.linspace(0,0.044,epx)
J0 = np.linspace(0,0.05,jpx)
J1 = np.linspace(0.05,0.15,jpx)
J2 = np.linspace(0.15,0.3,jpx)
p=1



#initialize the array

w0,v0=np.linalg.eigh(M(Del,D[0],J1[0],E[0],p)[0])
w1,v1=np.linalg.eigh(M(Del,D[0],J1[0],E[0],p)[1])

e_0 = np.zeros((totpx,len(w0)))*np.NaN
e_1 = np.zeros((totpx,len(w0)))*np.NaN

# calculate diagram

n = 0

for i in J0:
    w0,v0=np.linalg.eigh(M(Del,D[0],i,E[0],p)[0])
    w1,v1=np.linalg.eigh(M(Del,D[0],i,E[0],p)[1])
    for j in range(0,len(w0)):
        e_0[n,j] = w0[j]
        e_1[n,j] = w1[j]
    n += 1

for i in D:
    w0,v0=np.linalg.eigh(M(Del,i,J0[-1],E[0],p)[0])
    w1,v1=np.linalg.eigh(M(Del,i,J0[-1],E[0],p)[1])
    for j in range(0,len(w0)):
        e_0[n,j] = w0[j]
        e_1[n,j] = w1[j]
    n += 1

for i in J1:
    w0,v0=np.linalg.eigh(M(Del,D[-1],i,E[0],p)[0])
    w1,v1=np.linalg.eigh(M(Del,D[-1],i,E[0],p)[1])
    for j in range(0,len(w0)):
        e_0[n,j] = w0[j]
        e_1[n,j] = w1[j]

    n += 1


for i in E:
    w0,v0=np.linalg.eigh(M(Del,D[-1],J1[-1],i,p)[0])
    w1,v1=np.linalg.eigh(M(Del,D[-1],J1[-1],i,p)[1])
    for j in range(0,len(w0)):
        e_0[n,j] = w0[j]
        e_1[n,j] = w1[j]
    n += 1

for i in J2:
    w0,v0=np.linalg.eigh(M(Del,D[-1],i,E[-1],p)[0])
    w1,v1=np.linalg.eigh(M(Del,D[-1],i,E[-1],p)[1])
    for j in range(0,len(w0)):
        e_0[n,j] = w0[j]
        e_1[n,j] = w1[j]
    n += 1

#figure setup

plt.rcParams.update({'font.size': 12})
cm = 1/2.54
f = plt.figure(figsize=(10.5*cm,7*cm))
ax = []
gs1 = GridSpec(1, 5,width_ratios=[1,1,1,1,2],wspace=0)
ax.append(f.add_subplot(gs1[0,0]))
ax.append(f.add_subplot(gs1[0,1]))
ax.append(f.add_subplot(gs1[0,2]))
ax.append(f.add_subplot(gs1[0,3]))
ax.append(f.add_subplot(gs1[0,4]))


ax[1].yaxis.set_visible(False)
ax[2].yaxis.set_visible(False)
ax[3].yaxis.set_visible(False)
ax[4].yaxis.set_visible(False)

#plot

ax[0].plot(J0,e_0[:100,:len(w0)],color='C0')
ax[0].plot(J0,e_1[:100,:len(w0)],color='C1')
ax[1].plot(D,e_0[100:200,:len(w0)],color='C0')
ax[1].plot(D,e_1[100:200,:len(w0)],color='C1')
ax[2].plot(J1,e_0[200:300,:len(w0)],color='C0')
ax[2].plot(J1,e_1[200:300,:len(w0)],color='C1')
ax[3].plot(E*1e3,e_0[300:400,:len(w0)],color='C0')
ax[3].plot(E*1e3,e_1[300:400,:len(w0)],color='C1')
ax[4].plot(J2,e_0[400:500,:len(w0)],color='C0')
ax[4].plot(J2,e_1[400:500,:len(w0)],color='C1')


#graph settings

ax[0].set_xlim(0,J0[-1])
ax[0].set_xlabel('J '+r'($\Delta$)')
ax[0].set_ylabel('Energy '+r'($\Delta$)')
#ax[0].set_xticks([0,0.9])

ax[1].set_xlim(0,D[-1])
ax[1].set_xlabel('D '+r'($\Delta$)')
ax[1].set_ylabel('Energy '+r'($\Delta$)')
ax[1].set_xticks([0,0.9])
ax[2].set_xlim(J1[0],J1[-1])
ax[2].xaxis.tick_top()
ax[2].xaxis.set_label_position('top')
ax[2].set_xlabel('J '+r'($\Delta$)')
ax[3].set_xticks([0,np.round(J1[-1],2)])
ax[3].set_xlim(E[0]*1e3,E[-1]*1e3)
ax[3].set_xlabel('E '+r'$\cdot10^{-3}(\Delta)$')
ax[3].set_xticks([0,44])
ax[4].set_xlim(J2[0],J2[-1])
ax[4].xaxis.tick_top()
ax[4].set_xlabel('J '+r'($\Delta$)')
ax[4].xaxis.set_label_position('top')
ax[4].set_xticks([np.round(J2[0],2),np.round(J2[-1],2)])

for axn in ax:
    axn.set_ylim(-1.3,7)
    axn.tick_params(axis='x',direction='in')
    axn.tick_params(axis='y',direction='in')


plt.tight_layout()

plt.savefig('C:/Users/strivini/Desktop/scheme.pdf',dpi=600)