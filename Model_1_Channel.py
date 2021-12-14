import numpy as np
import matplotlib.pyplot as plt
import nanonis


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
Del=0.7
kJ1 = 1
cal = 1.0312
sub_ing = 0.5
cmap = 'magma'
Deltat = 0.68
a = np.array([0.596899091,0.592263,0.599925,0.592789,0.590584,0.597909,0.634405,0.648816,0.673330,0.72108,0.759739,0.781286,0.782148,0.778115,0.762544,0.745005,0.696710,0.652128,0.616709,0.606191,0.593855,0.590836,0.596292,0.606088,0.609632,])
a = a*cal
a = a-Deltat
J1 = kJ1*(Del-a)*4/7
p=1



En = np.linspace(-1,3,1000)


D=0.7
E=0
c = 0
Deff = D*(1-c*J1**2)



def energyCalc(Del,D,J1,p,E,En):
    peakn = 6
    #initialize the array
    w0,v0=np.linalg.eigh(M(Del,D[0],J1[0],E,p)[0])
    w1,v1=np.linalg.eigh(M(Del,D[0],J1[0],E,p)[1])
    peaks_weak = np.zeros((len(J1),peakn))
    peaks_strong = np.zeros((len(J1),peakn))
    count = 0
    for i in range(len(J1)):
        w0,v0=np.linalg.eigh(M(Del,D[i],J1[i],E,p)[0])
        w1,v1=np.linalg.eigh(M(Del,D[i],J1[i],E,p)[1])
        for j in range(peakn):
            peaks_weak[i,j] = w1[j]-min(w0)
        for j in range(peakn):
            peaks_strong[i,j] = w0[j]-min(w1)
    return peaks_weak, peaks_strong



f1, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)


#LINESCAN PLOT
filenames80 = []
frm1 = np.linspace(80,99,20)
frm2 = np.linspace(100,104,5)
for i in frm1:
    filenames80.append('C:/Users/jonor/Documents/GitHub/linscan/CS04200823_0{}.dat'.format(int(i)))
for i in frm2:
    filenames80.append('C:/Users/jonor/Documents/GitHub/linscan/CS04200823_{}.dat'.format(int(i)))

map80 = nanonis.linescan()
map80.load(filenames80)
#map80.hand_normalization('hand_norm80.txt')
map80.biasCalibration(cal)
map80.conductance = np.roll(map80.conductance,-1,1)

#tp subpress ingap conductance

map80.conductance[:,72:227] = map80.conductance[:,72:227]*sub_ing

ax.imshow(np.fliplr(map80.conductance),aspect='auto',extent=[min(map80.bias*1e3),max(map80.bias*1e3),min(map80.distance),max(map80.distance)],cmap = cmap,interpolation='nearest',vmin=-0.21,vmax=1)



peaks_weak, peaks_strong = energyCalc(Del,Deff,J1,p,E,En)

for i in range(0,peaks_weak.shape[1]):
    ax.plot(Deltat+peaks_weak[:,i],   map80.distance,color='C3')
    ax.plot(Deltat+peaks_strong[:,i], map80.distance,color='C2')
    ax.plot(-Deltat-peaks_weak[:,i] , map80.distance,color='C3')
    ax.plot(-Deltat-peaks_strong[:,i],map80.distance,color='C2')

# ax.set_xlim(map80.bias[0]*1e3,map80.bias[-1]*1e3)

# # sliders

ac = f1.add_axes([0.1, 0.1, 0.65, 0.03])
aD = f1.add_axes([0.1, 0.15, 0.65, 0.03])
akJ1 = f1.add_axes([0.1, 0.20, 0.65, 0.03])
aE = f1.add_axes([0.1, 0.05, 0.65, 0.03])
ap = f1.add_axes([0.1, 0.25, 0.65, 0.03])



 
c = plt.Slider(ac,'c',-2,2,valinit=0)
D = plt.Slider(aD, 'D', 0, 5, valinit =0.7)
kJ1 = plt.Slider(akJ1, 'J1', -2, 2, valinit =1)
E = plt.Slider(aE, 'E', -0.1, 0.1, valinit =0.0)
p = plt.Slider(ap, 'p', -3, 3, valinit =1)

def update(val):
    ax.clear()
    Deltat = 0.68
    a = np.array([0.596899091,0.592263,0.599925,0.592789,0.590584,0.597909,0.634405,0.648816,0.673330,0.72108,0.759739,0.781286,0.782148,0.778115,0.762544,0.745005,0.696710,0.652128,0.616709,0.606191,0.593855,0.590836,0.596292,0.606088,0.609632,])
    a = a*1.0312
    a = a-Deltat
    J1= kJ1.val*(Del-a)*4/7
    Deff = D.val*(1-c.val*J1**2)
    b=np.linspace(0,1,len(a))
    ax.imshow(np.fliplr(map80.conductance),aspect='auto',extent=[min(map80.bias*1e3),max(map80.bias*1e3),min(map80.distance),max(map80.distance)],cmap = cmap,interpolation='nearest',vmin=-0.21,vmax=1)
    peaks_weak, peaks_strong = energyCalc(Del,Deff,J1,p.val,E.val,En)
    for i in range(0,peaks_weak.shape[1]):
        ax.plot(Deltat+peaks_weak[:,i],   map80.distance,color='C3')
        ax.plot(Deltat+peaks_strong[:,i], map80.distance,color='C2')
        ax.plot(-Deltat-peaks_weak[:,i] , map80.distance,color='C3')
        ax.plot(-Deltat-peaks_strong[:,i],map80.distance,color='C2')
    # ax.set_xlim(map80.bias[0]*1e3,map80.bias[-1]*1e3)

D.on_changed(update)
kJ1.on_changed(update)
E.on_changed(update)
c.on_changed(update)
p.on_changed(update)

plt.show()