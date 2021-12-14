import numpy as np
import matplotlib.pyplot as plt
import nanonis


def M(Del,D,J1,J2):
    Delta= [[Del, 0, 0, 0, 0, 0], [0, Del, 0, 0, 0, 0], [0, 0, Del, 0, 0, 0], [0, 0, 0, Del, 0, 0], [0, 0, 0, 0, Del, 0], [0, 0, 0, 0, 0, Del]]
    Spin= [[(25/4)*D, 0, 0, 0, 0, 0], [0, (9/4)*D, 0, 0, 0, 0], [0, 0, (1/4)*D, 0, 0, 0], [0, 0, 0, (1/4)*D, 0, 0], [0, 0, 0, 0, (9/4)*D, 0], [0, 0, 0, 0, 0, (25/4)*D]]
    DJ1= [[(5/4)*J1, 0, 0, 0, 0, 0], [0, (3/4)*J1, 0, 0, 0, 0], [0, 0, (1/4)*J1, 0, 0, 0], [0, 0, 0, -(1/4)*J1, 0, 0], [0, 0, 0, 0, -(3/4)*J1, 0], [0, 0, 0, 0, 0, -(5/4)*J1]]
    ND1J1= [[0, 0, 0, 0, 0, 0], [(np.sqrt(5)/2)*J1, 0, 0, 0, 0, 0], [0, np.sqrt(2)*J1, 0, 0, 0, 0], [0, 0, (3/2)*J1, 0, 0, 0], [0, 0, 0, np.sqrt(2)*J1, 0, 0], [0, 0, 0, 0, (np.sqrt(5)/2)*J1, 0]]
    ND2J1= [[0, (np.sqrt(5)/2)*J1, 0, 0, 0, 0], [0, 0, np.sqrt(2)*J1, 0, 0, 0], [0, 0, 0, (3/2)*J1, 0, 0], [0, 0, 0, 0, np.sqrt(2)*J1, 0], [0, 0, 0, 0, 0, (np.sqrt(5)/2)*J1], [0, 0, 0, 0, 0, 0]];
    DJ2= [[(5/4)*J2, 0, 0, 0, 0, 0], [0, (3/4)*J2, 0, 0, 0, 0], [0, 0, (1/4)*J2, 0, 0, 0], [0, 0, 0, -(1/4)*J2, 0, 0], [0, 0, 0, 0, -(3/4)*J2, 0], [0, 0, 0, 0, 0, -(5/4)*J2]]
    ND1J2= [[0, 0, 0, 0, 0, 0], [(np.sqrt(5)/2)*J2, 0, 0, 0, 0, 0], [0, np.sqrt(2)*J2, 0, 0, 0, 0], [0, 0, (3/2)*J2, 0, 0, 0], [0, 0, 0, np.sqrt(2)*J2, 0, 0], [0, 0, 0, 0, (np.sqrt(5)/2)*J2, 0]]
    ND2J2= [[0, (np.sqrt(5)/2)*J2, 0, 0, 0, 0], [0, 0, np.sqrt(2)*J2, 0, 0, 0], [0, 0, 0, (3/2)*J2, 0, 0], [0, 0, 0, 0, np.sqrt(2)*J2, 0], [0, 0, 0, 0, 0, (np.sqrt(5)/2)*J2], [0, 0, 0, 0, 0, 0]]
    tau0 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    tauz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]
    tauzz = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]
    tau1 = [[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]]
    tau21 = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]]
    tau22 = [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
    tau31 = [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    tau32 = [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]]
    Mat11= np.kron(tau0, Spin)+np.kron(tau1, Delta)
    Mat01= np.kron(tau0, Spin)+np.kron(tauz, DJ1)+np.kron(np.add(tau21,tau22), Delta)+np.kron(tau31, ND1J1)+np.kron(tau32, ND2J1)
    Mat10= np.kron(tau0, Spin)+np.kron(tauz, DJ2)+np.kron(np.add(tau21,tau22), Delta)+np.kron(tau31, ND1J2)+np.kron(tau32, ND2J2)
    Mat00= np.kron(tau0, Spin) + np.kron(tauz, DJ1) + np.kron(tauzz, DJ2) + np.kron(tau31, ND1J1) + np.kron(tau32, ND2J1) + np.kron(tau21, ND1J2) + np.kron(tau22, ND2J2)
    return [Mat11,Mat01,Mat10,Mat00]

    
def peak(En,Gamma,DeltaE,w):
    return w*np.divide(Gamma,np.add(np.power(np.add(En,-DeltaE),2),Gamma**2/2))

ES1=[0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0,0, 0]
ES2=[0,0,0,-1,0,0,0,0,1,0,0,0,0,0,+1,0,0,0,0,0,-1,0,0,0]
ES3=[0,0,1,1,0,0,0,0,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]



Del=0.7

kJ1 = 1
kJ2 = 1
cal = 1.0312
sub_ing = 0.5
cmap = 'magma'



Deltat = 0.68
a = np.array([0.596899091,0.592263,0.599925,0.592789,0.590584,0.597909,0.634405,0.648816,0.673330,0.72108,0.759739,0.781286,0.782148,0.778115,0.762544,0.745005,0.696710,0.652128,0.616709,0.606191,0.593855,0.590836,0.596292,0.606088,0.609632,])
a = a*1.0312
a = a-Deltat
J1 = kJ1*(Del-a)*4/7

J2 = kJ2*(Del-a)*4/7


T1=1
T2=0.2


En = np.linspace(0,3,1000)


D=0.7
# c = 1.298
# Deff = D*(1-c*J**2)



def energyCalc(Del,D,J1,J2,En):
    peakn=6
    peaks_weak1 = np.zeros((len(J1),peakn))
    peaks_weak2 = np.zeros((len(J1),peakn))
    peaks_strong1 = np.zeros((len(J1),peakn))
    peaks_strong2 = np.zeros((len(J1),peakn))
    for i in range(len(J1)):
        w0,v0=np.linalg.eigh(M(Del,D,J1[i],J2[i])[0])
        w1,v1=np.linalg.eigh(M(Del,D,J1[i],J2[i])[1])
        w2,v2=np.linalg.eigh(M(Del,D,J1[i],J2[i])[2])
        w3,v3=np.linalg.eigh(M(Del,D,J1[i],J2[i])[3])
        for j in range(peakn):
            peaks_weak1[i,j]=w1[j]-min(w0)

            peaks_weak2[i,j]=w2[j]-min(w0)

            peaks_strong1[i,j]=w0[j]-min(w1)

            peaks_strong2[i,j]=w3[j]-min(w1)
    return  peaks_weak1, peaks_weak2, peaks_strong1, peaks_strong2



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


peaks_weak1,peaks_weak2, peaks_strong1, peaks_strong2 = energyCalc(Del,D,J1,J2,En)

for i in range(0,peaks_weak1.shape[1]):
    ax.plot(Deltat+peaks_weak1[:,i],   map80.distance,color='C3')
    ax.plot(Deltat+peaks_weak2[:,i],   map80.distance,color='C3')
    ax.plot(Deltat+peaks_strong1[:,i], map80.distance,color='C2')
    ax.plot(Deltat+peaks_strong2[:,i], map80.distance,color='C2')
    ax.plot(-Deltat-peaks_weak1[:,i] , map80.distance,color='C3')
    ax.plot(-Deltat-peaks_weak2[:,i] , map80.distance,color='C3')
    ax.plot(-Deltat-peaks_strong1[:,i],map80.distance,color='C2')
    ax.plot(-Deltat-peaks_strong2[:,i],map80.distance,color='C2')


# sliders

# ac = f1.add_axes([0.25, 0.1, 0.65, 0.03])
aD = f1.add_axes([0.1, 0.15, 0.65, 0.03])
akJ1 = f1.add_axes([0.1, 0.10, 0.65, 0.03])
akJ2 = f1.add_axes([0.1, 0.05, 0.65, 0.03])
#aT1 = f1.add_axes([0.1, 0.20, 0.65, 0.03])
#aT2 = f1.add_axes([0.1, 0.25, 0.65, 0.03])





 

D = plt.Slider(aD, 'D', 0, 5, valinit =1.298)
kJ1 = plt.Slider(akJ1, 'J1', -2, 2, valinit =1)
kJ2 = plt.Slider(akJ2, 'J2', -2, 2, valinit =1)
#T1 = plt.Slider(aT1, 'T1', -3, 3, valinit =0.822)
#T2 = plt.Slider(aT2, 'T2', -3, 3, valinit =0.822)



def update(val):
    ax.clear()

    Deltat = 0.68
    a = np.array([0.596899091,0.592263,0.599925,0.592789,0.590584,0.597909,0.634405,0.648816,0.673330,0.72108,0.759739,0.781286,0.782148,0.778115,0.762544,0.745005,0.696710,0.652128,0.616709,0.606191,0.593855,0.590836,0.596292,0.606088,0.609632,])
    a = a*1.0312
    a = a-Deltat
    J1 = kJ1.val*(Del-a)*4/7

    J2 = kJ2.val*(Del-a)*4/7

    peaks_weak1,peaks_weak2, peaks_strong1, peaks_strong2 = energyCalc(Del,D.val,J1,J2,En)
    ax.imshow(np.fliplr(map80.conductance),aspect='auto',extent=[min(map80.bias*1e3),max(map80.bias*1e3),min(map80.distance),max(map80.distance)],cmap = cmap,interpolation='nearest',vmin=-0.21,vmax=1)
    for i in range(0,peaks_weak1.shape[1]):
        ax.plot(Deltat+peaks_weak1[:,i],   map80.distance,color='C3')
        ax.plot(Deltat+peaks_weak2[:,i],   map80.distance,color='C2')
        ax.plot(Deltat+peaks_strong1[:,i], map80.distance,color='C3')
        ax.plot(Deltat+peaks_strong2[:,i], map80.distance,color='C2')
        ax.plot(-Deltat-peaks_weak1[:,i] , map80.distance,color='C3')
        ax.plot(-Deltat-peaks_weak2[:,i] , map80.distance,color='C3')
        ax.plot(-Deltat-peaks_strong1[:,i],map80.distance,color='C2')
        ax.plot(-Deltat-peaks_strong2[:,i],map80.distance,color='C2')

D.on_changed(update)
kJ1.on_changed(update)
kJ2.on_changed(update)
#T1.on_changed(update)
#T2.on_changed(update)






plt.show()

