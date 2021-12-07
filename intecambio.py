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

def thermal(E1,E2,E,beta):
    a=np.exp(-E1*beta)+np.exp(-E2*beta)
    return np.exp(-E*beta)/a

cpup=np.kron([[0,1],[0,0]],np.identity(6))
cpdown01=np.kron([[0,0],[0,1]],np.identity(6))
cpdown10=np.kron([[1,0],[0,0]],np.identity(6))

Del=0.7


kJ1 = 1







En = np.linspace(-1,3,1000)

J=np.linspace(0,0.37,100)
D=0
U=0
# c = 1.298
# Deff = D*(1-c*J**2)



y=[]
yy=[]
for i in J:
    w0,v0=np.linalg.eigh(M(Del,D,i,U)[0])
    w1,v1=np.linalg.eigh(M(Del,D,i,U)[1])
    for j in range(len(w0)):
            plt.plot(i,w0[j],'bo',markersize=1)
            plt.plot(i,w1[j],'ro',markersize=1)
J=0.37
#U=np.concatenate((np.linspace(0.0,0.1,50),np.linspace(0.1,0,50)))
U=np.linspace(0.0,0.1,100)

for i in range(len(U)):
    w0,v0=np.linalg.eigh(M(Del,D,J,U[i])[0])
    w1,v1=np.linalg.eigh(M(Del,D,J,U[i])[1])
    for j in range(len(w0)):
        plt.plot(i*0.0025+0.38,w0[j],'bo')
        plt.plot(i*0.0025+0.38,w1[j],'ro')

print(v1[:,1],v1[:,2])

#J=np.linspace(0.37, 0.6,100)
#U=0
#
#for i in J:
#    w0,v0=np.linalg.eigh(M(Del,D,i,U)[0])
#    w1,v1=np.linalg.eigh(M(Del,D,i,U)[1])
#    for j in range(len(w0)):
#        temp1=np.dot(v1[:,j],np.dot(cpup,v0[:,0]))
#        if np.abs(temp1)>0.000001:
#            plt.plot(i+0.47,w0[j],'bo')
#    for j in range(len(w1)):
#        temp1=np.dot(v0[:,j],np.dot(cpup,v1[:,0]))
#        if np.abs(temp1)>0.000001:
#            plt.plot(i+0.47,w1[j],'ro')

plt.show()