import numpy as np
import matplotlib.pyplot as plt


def M(Del,D,J,U):
    Mat1=[[(25/4)*D + (5/4)*J, 0, np.sqrt(10)*U, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
    [0, (9/4)*D + (3/4)*J, 0, 3*np.sqrt(2)*U, 0, 0, (J/2)*np.sqrt(5), 0, 0, 0, 0, 0],
    [np.sqrt(10)*U, 0, (1/4)*D + (1/4)*J, 0, 3*np.sqrt(2)*U, 0, 0, J*np.sqrt(2), 0, 0, 0, 0], 
    [0, 3*np.sqrt(2)*U, 0, (1/4)*D - (1/4)*J, 0, np.sqrt(10)*U, 0, 0, (J/2)*3, 0, 0, 0], 
    [0, 0, 3*np.sqrt(2)*U, 0, (9/4)*D - (3/4)*J, 0, 0, 0, 0, J*np.sqrt(2), 0, 0], 
    [0, 0, 0, np.sqrt(10)*U, 0, (25/4)*D - (5/4)*J, 0, 0, 0, 0, (J/2)*np.sqrt(5), 0], 
    [0, (J/2)*np.sqrt(5), 0, 0, 0, 0, (25/4)*D - (5/4)*J, 0, np.sqrt(10)*U, 0, 0, 0], 
    [0, 0, J*np.sqrt(2), 0, 0, 0, 0, (9/4)*D - (3/4)*J, 0, 3*np.sqrt(2)*U, 0, 0], 
    [0, 0, 0, (3*J/2), 0, 0, np.sqrt(10)*U, 0, (1/4)*D - (1/4)*J, 0, 3*np.sqrt(2)*U, 0], 
    [0, 0, 0, 0, J*np.sqrt(2), 0, 0, 3*np.sqrt(2)*U, 0, (1/4)*D + (1/4)*J, 0, np.sqrt(10)*U], 
    [0, 0, 0, 0, 0, (J/2)*np.sqrt(5), 0, 0, 3*np.sqrt(2)*U, 0, (9/4)*D + (3/4)*J, 0], 
    [0, 0, 0, 0, 0, 0, 0, 0, 0, np.sqrt(10)*U, 0, (25/4)*D + (5/4)*J]]
    Mat0=[[(25/4)*D, 0, np.sqrt(10)*U, 0, 0, 0, Del, 0, 0, 0, 0, 0], 
    [0, (9/4)*D, 0, 3*np.sqrt(2)*U, 0, 0, 0, Del, 0, 0, 0, 0], 
    [np.sqrt(10)*U, 0, (1/4)*D, 0, 3*np.sqrt(2)*U, 0, 0, 0, Del, 0, 0, 0], 
    [0, 3*np.sqrt(2)*U, 0, (1/4)*D, 0, np.sqrt(10)*U, 0, 0, 0, Del, 0, 0], 
    [0, 0, 3*np.sqrt(2)*U, 0, (9/4)*D, 0, 0, 0, 0, 0, Del, 0], 
    [0, 0, 0, np.sqrt(10)*U, 0, (25/4)*D, 0, 0, 0, 0, 0, Del], 
    [Del, 0, 0, 0, 0, 0, (25/4)*D, 0, np.sqrt(10)*U, 0, 0, 0], 
    [0, Del, 0, 0, 0, 0, 0, (9/4)*D, 0, 3*np.sqrt(2)*U, 0, 0], 
    [0, 0, Del, 0, 0, 0, np.sqrt(10)*U, 0, (1/4)*D, 0, 3*np.sqrt(2)*U, 0], 
    [0, 0, 0, Del, 0, 0, 0, 3*np.sqrt(2)*U, 0, (1/4)*D, 0, np.sqrt(10)*U], 
    [0, 0, 0, 0, Del, 0, 0, 0, 3*np.sqrt(2)*U, 0, (9/4)*D, 0], 
    [0, 0, 0, 0, 0, Del, 0, 0, 0, np.sqrt(10)*U, 0, (25/4)*D]]
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

Del=0.7


kJ1 = 1



Deltat = 0.68
a = np.array([0.596899091,0.592263,0.599925,0.592789,0.590584,0.597909,0.634405,0.648816,0.673330,0.72108,0.759739,0.781286,0.782148,0.778115,0.762544,0.745005,0.696710,0.652128,0.616709,0.606191,0.593855,0.590836,0.596292,0.606088,0.609632,])
a = a*1.0312
a = a-Deltat
#b=np.linspace(0,1,len(a))

#a=np.interp(np.linspace(0,1,200),b,a)
J1 = kJ1*(Del-a)*4/7




En = np.linspace(-1,3,1000)


D=0.7
U=0.05
# c = 1.298
# Deff = D*(1-c*J**2)



def energyCalc(Del,D,J1,U,En):
    Y=[]
    for i in range(len(J1)):
        y=[]
        for aaa in En:
            y.append(0)
        w0,v0=np.linalg.eigh(M(Del,D,J1[i],U)[0])
        w1,v1=np.linalg.eigh(M(Del,D,J1[i],U)[1])
        for j in range(len(w0)):
            temp1=np.dot(v1[:,j],np.dot(cpup,v0[:,0]))
            if np.abs(temp1)>0.0001:
                y=np.add(y,peak(En,0.01,w1[j]-min(w0),thermal(min(w0),min(w1),min(w0))))
            temp2=np.dot(v0[:,j],np.dot(cpup,v1[:,0]))
            if np.abs(temp2)>0.0001:
                y=np.add(y,peak(En,0.01,w0[j]-min(w1),thermal(min(w0),min(w1),min(w1))))

        Y.append(y/np.linalg.norm(y))
    return Y



f1, ax = plt.subplots()
plt.subplots_adjust(bottom=0.4)


Y = energyCalc(Del,D,J1,U,En)

ax.imshow(Y,aspect='auto',interpolation='nearest',extent=(-1,3,0,1))
# sliders

# ac = f1.add_axes([0.25, 0.1, 0.65, 0.03])
aD = f1.add_axes([0.1, 0.15, 0.65, 0.03])
akJ1 = f1.add_axes([0.1, 0.10, 0.65, 0.03])
aU = f1.add_axes([0.1, 0.05, 0.65, 0.03])




 

D = plt.Slider(aD, 'D', 0, 5, valinit =0.7)
kJ1 = plt.Slider(akJ1, 'J1', -2, 2, valinit =1)
U = plt.Slider(aU, 'U', -2, 2, valinit =0.05)


def update(val):
    ax.clear()

    Deltat = 0.68
    a = np.array([0.596899091,0.592263,0.599925,0.592789,0.590584,0.597909,0.634405,0.648816,0.673330,0.72108,0.759739,0.781286,0.782148,0.778115,0.762544,0.745005,0.696710,0.652128,0.616709,0.606191,0.593855,0.590836,0.596292,0.606088,0.609632,])
    a = a*1.0312
    a = a-Deltat
    b=np.linspace(0,1,len(a))


    a=np.interp(np.linspace(0,1,200),b,a)
    J1 = kJ1.val*(Del-a)*4/7



    Y = energyCalc(Del,D.val,J1,U.val,En)
    ax.imshow(Y,aspect='auto',interpolation='nearest',extent=(-1,3,0,1))

D.on_changed(update)
kJ1.on_changed(update)
U.on_changed(update)

plt.show()