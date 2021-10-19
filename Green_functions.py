import numpy as np
import scipy.special as scisp
import scipy.integrate as integ
import mpmath
from scipy import constants as const
##  THIS PROGRAME IS A MODULE TO CALCULATE THE EXPRESSION FOR BCS REAL SPACE GREEN'S FUNCTIONS ##
##  IT ALSO CALCULATES THE PERTURBED SHIBA STATE GREEN'S FUNCTIONS ##


##  THIS PROGRAM DOES NOT PLOT ANYTHING. ANOTHER PROGRAMM IS NEEDED TO PLOT ANY KIND OF DOS, SPDOS ... ##

print('hello world')

class Green:

    def __init(self):
        pass
        # super(Green, self).__init__

    def increasePoints(self, number):
        pass

    def G(self,x1,x2,En,delta,m,pf,mode):
        ## x1 and x2 are the spatial coordinates. They need to be points, not arrays ##
        ## En must be a single complex number ##
        #define the BCS dispersion relation 
        w=np.sqrt( np.power(delta,2)-np.power(En,2))
        #pauli matrices
        tau00=np.identity(4)
        tau10=np.array( [ [0,0,1,0] , [0,0,0,1] , [1,0,0,0] , [0,1,0,0] ] )
        tau30=np.array( [ [1,0,0,0] , [0,1,0,0] , [0,0,-1,0] , [0,0,0,-1] ] )

        BCS=np.divide(En,w)*tau00+np.divide(delta,w)*tau10
        xi=tau30
        if mode==0:
            ## mode=0 is for the spherical Fermi surface ##
            ## for this mode x1 is the radial distance and x2 is the angle ##
            u=np.complex(x1*pf,x1*m*w/(pf))
            a=scisp.jv(0,u) #bessel function 0th order u=argument
            b=mpmath.struveh(0,u) #gives the struve function
            self.G0=-m/2*(np.complex(np.real(a+np.complex(0,1)*b),0))*BCS+m/2*(np.complex(np.imag(a+np.complex(0,1)*b),0))*xi

        if mode==1:
            ## mode=1 is for the square-like fermi surface ##
            ## for this mode x1=x and x2=y ##
            if np.abs(x1)<0.00001 and np.abs(x2)<0.00001:
                G1=(2*m/np.pi)*BCS
                G2=0.0
            elif np.abs(x2)<0.00001:
                G1=(np.exp(-(m*w/pf)*np.abs(x1))*( (1/np.abs(x1))*pf*np.sin(pf*(np.abs(x1)))+(pf**2)*np.cos(pf*np.abs(x1)) ) )*(m/(np.pi*pf**2))*BCS
                G2=(np.exp(-(m*w/pf)*np.abs(x1))*( (1/np.abs(x1))*pf*np.cos(pf*(np.abs(x1)))+(pf**2)*np.sin(pf*np.abs(x1)) ) -pf/np.abs(x1))*(m/(np.pi*pf**2))*xi
            elif np.abs(x1)<0.00001:
                G1=(np.exp(-(m*w/pf)*np.abs(x2))*( (1/np.abs(x2))*pf*np.sin(pf*(np.abs(x2)))+(pf**2)*np.cos(pf*np.abs(x2)) ) )*(m/(np.pi*pf**2))*BCS
                G2=(np.exp(-(m*w/pf)*np.abs(x2))*( (1/np.abs(x2))*pf*np.cos(pf*(np.abs(x2)))+(pf**2)*np.sin(pf*np.abs(x2)) ) -pf/np.abs(x2))*(m/(np.pi*pf**2))*xi
            
            
            elif x1+x2>=0.00001 and x1-x2>=0.00001:
                G1=(np.exp(-(m*w/pf)*(x1+x2))*(1/x1+1/x2)*pf*np.sin(pf*(x1+x2))+np.exp(-(m*w/pf)*(x1-x2))*(1/x1-1/x2)*pf*np.sin(pf*(x1-x2)) )*(m/(2*np.pi*pf**2))*BCS
                G2=(np.exp(-(m*w/pf)*(x1+x2))*(1/x1+1/x2)*pf*np.cos(pf*(x1+x2))+np.exp(-(m*w/pf)*(x1-x2))*(1/x1-1/x2)*pf*np.cos(pf*(x1-x2)) -2*pf/x1)*(m/(2*np.pi*pf**2))*xi
            elif x1+x2>0.00001 and x1-x2<0.00001:
                G1=(np.exp(-(m*w/pf)*(x1+x2))*(1/x1+1/x2)*pf*np.sin(pf*(x1+x2))+np.exp(-(m*w/pf)*(x2-x1))*(1/x2-1/x1)*pf*np.sin(pf*(x2-x1)) )*(m/(2*np.pi*pf**2))*BCS
                G2=(np.exp(-(m*w/pf)*(x1+x2))*(1/x1+1/x2)*pf*np.cos(pf*(x1+x2))+np.exp(-(m*w/pf)*(x2-x1))*(1/x2-1/x1)*pf*np.cos(pf*(x2-x1)) -2*pf/x2)*(m/(2*np.pi*pf**2))*xi
            elif x1+x2<=0.00001 and x1-x2<=0.00001:
                G1=(np.exp((m*w/pf)*(x1+x2))*(-1/x1-1/x2)*pf*np.sin(pf*(-x1-x2))+np.exp(-(m*w/pf)*(-x1+x2))*(-1/x1+1/x2)*pf*np.sin(pf*(-x1+x2)) )*(m/(2*np.pi*pf**2))*BCS
                G2=(np.exp((m*w/pf)*(x1+x2))*(-1/x1-1/x2)*pf*np.cos(pf*(-x1-x2))+np.exp(-(m*w/pf)*(-x1+x2))*(-1/x1+1/x2)*pf*np.cos(pf*(-x1+x2)) +2*pf/x1)*(m/(2*np.pi*pf**2))*xi
            elif x1+x2<0.00001 and x1-x2>0.00001:
                G1=(np.exp((m*w/pf)*(x1+x2))*(-1/x1-1/x2)*pf*np.sin(pf*(-x1-x2))+np.exp(-(m*w/pf)*(-x2+x1))*(-1/x2+1/x1)*pf*np.sin(pf*(-x2+x1)) )*(m/(2*np.pi*pf**2))*BCS
                G2=(np.exp((m*w/pf)*(x1+x2))*(-1/x1-1/x2)*pf*np.cos(pf*(-x1-x2))+np.exp(-(m*w/pf)*(-x2+x1))*(-1/x2+1/x1)*pf*np.cos(pf*(-x2+x1)) +2*pf/x2)*(m/(2*np.pi*pf**2))*xi
            self.G0=-G1-G2

        if mode==2:
            ## mode=2 is for the hexagonal-shaped Fermi contour
            ## x1=x and x2=y
            R=[[-1/2,-np.sqrt(3)/2],[np.sqrt(3)/2,-1/2]]
            if x2>0 and x2>np.sqrt(3)*x1:
                D=np.matmul(np.matmul(R,R),[x1,x2])
                x1=D[0]
                x2=D[1]
            elif x2<0 and x2<-np.sqrt(3)*x1:
                D=np.matmul(R,[x1,x2])
                x1=D[0]
                x2=D[1]
            chi=m*w/pf
            a=np.divide(2,np.sqrt(3))*np.abs(x2)
            b=x1-np.divide(1,np.sqrt(3))*np.abs(x2)
            c=x1+np.divide(1,np.sqrt(3))*np.abs(x2)
            if np.abs(x1) < 0.0001 and np.abs(x2) < 0.0001:
                G1 = np.sqrt(3)*pf**2
                G2 = 0.0
            elif np.abs(x2)<0.001:
                G1=(2*pf/(np.sqrt(3)*x1))*np.exp(-chi*x1)*np.sin(pf*x1)+(np.cos(pf*x1)/np.sqrt(3))*np.exp(-chi*x1)
                G2=-(2*pf/(np.sqrt(3)*x1))*np.exp(-chi*x1)*np.cos(pf*x1)+(np.sin(pf*x1)/np.sqrt(3))*np.exp(-chi*x1)-1/(np.sqrt(3)*x1)
            elif np.abs(np.sqrt(3)*x1-np.abs(x2))<0.001:
                G1=(pf/(np.sqrt(3)*x1))*np.exp(-2*chi*x1)*np.sin(2*pf*x1)+(np.cos(2*pf*x1)/np.sqrt(3))*np.exp(-2*chi*x1)
                G2=-(pf/(np.sqrt(3)*x1))*np.exp(-2*chi*x1)*np.cos(2*pf*x1)+(np.sin(2*pf*x1)/np.sqrt(3))*np.exp(-2*chi*x1)-1/(2*np.sqrt(3)*x1)
            else:
                G1=np.exp(-chi*c)*np.sin(pf*c)*(1/(np.sqrt(3)*b)+1/(np.sqrt(3)*a))+np.exp(-chi*b)*np.sin(pf*b)*(-1/(np.sqrt(3)*a)+1/(np.sqrt(3)*c))+np.exp(-chi*a)*np.sin(pf*a)*(-1/(np.sqrt(3)*b)+1/(np.sqrt(3)*c))
                G2=np.exp(-chi*c)*np.cos(pf*c)*(-1/(np.sqrt(3)*b)-1/(np.sqrt(3)*a))+np.exp(-chi*b)*np.cos(pf*b)*(1/(np.sqrt(3)*a)-1/(np.sqrt(3)*c))+np.exp(-chi*a)*np.cos(pf*a)*(1/(np.sqrt(3)*b)-1/(np.sqrt(3)*c))-1/(np.sqrt(3)*c)
            self.G0=-(m/np.sqrt(3)*np.pi*pf**2)*G1*BCS-(m/np.sqrt(3)*np.pi*pf**2)*G2*xi
        
        
    def definitions(self,En,x,y,J1,J2,alpha,delta,m,pf,mode):
        ## The first atom is always at zero. The second atom coordinates are x and y ##
        ## x and y are cartesian coordinates for mode=1 and spherical coordenates for mode=0 ##
        self.A=np.array([[J1,0,0,0],[0,-J1,0,0],[0,0,J1,0],[0,0,0,-J1]])
        self.B=np.multiply(np.array([[J2,0,0,0],[0,-J2,0,0],[0,0,J2,0],[0,0,0,-J2]]),np.cos(alpha))+np.multiply(np.array([[0,J2,0,0],[J2,0,0,0],[0,0,0,J2],[0,0,J2,0]]),np.sin(alpha))
        self.G(0,0,En,delta,m,pf,mode)
        self.eta11= np.dot(self.G0,self.A)
        self.eta22= np.dot(self.G0,self.B)

        self.G(x,y,En,delta,m,pf,mode)
        self.eta12=np.dot(self.G0,self.B)
        self.eta21=np.dot(self.G0,self.A)

        self.temp1=np.linalg.inv(np.identity(4)-self.eta11)
        self.temp2=np.linalg.inv(np.identity(4)-self.eta22)
        self.F1=np.linalg.inv(np.identity(4)- np.dot( np.dot(self.eta12,self.temp2) ,self.eta21 ) -self.eta11   )
        self.F2=np.linalg.inv(np.identity(4)- np.dot( np.dot(self.eta21,self.temp1) ,self.eta12 ) -self.eta22   )
        self.zeta11=self.F1
        self.zeta22=self.F2
        self.zeta12=np.dot(self.F1,np.dot(self.eta12,self.temp2 ))
        self.zeta21=np.dot(self.F2,np.dot(self.eta21,self.temp1 ))
        
    def dG(self,x,y,En,x_atom,y_atom,J1,J2,alpha,delta,m,pf,mode): #dimer GF?
        ## The first atom is always at zero. The second atom coordinates are x_atom and y_atom ##
        ## x,y and x_atom and y_atom are cartesian coordinates for mode=1 and spherical coordenates for mode=0 ##
        if mode==0: #espherica
            self.G(x,y,En,delta,m,pf,0)
            tempg1=self.G0
            #convert in shperic coord
            R=np.sqrt(x**2+x_atom**2-2*x*x_atom*np.cos(y-y_atom))
            Phi=np.arctan((x*np.sin(y)-x_atom*np.sin(y_atom))/(x*np.cos(y)-x_atom*np.cos(y_atom)))
            self.G(R,Phi,En,delta,m,pf,0)
            tempg2=self.G0
            self.definitions(En,x_atom,y_atom,J1,J2,alpha,delta,m,pf,0)
            self.deltaG=np.dot(np.dot(tempg1,self.A ),np.dot(self.zeta11,tempg1) + np.dot(self.zeta12,tempg2) )+ np.dot( np.dot( tempg2 , self.B) , np.dot(self.zeta22,tempg2) + np.dot(self.zeta21,tempg1 ) )
        
        if mode==1: #cartesiana
            self.G(x,y,En,delta,m,pf,1)
            tempg1=self.G0
            self.G(x-x_atom,y-y_atom,En,delta,m,pf,1)
            tempg2=self.G0
            self.definitions(En,x_atom,y_atom,J1,J2,alpha,delta,m,pf,1)
            self.deltaG=np.dot(np.dot(tempg1,self.A ),np.dot(self.zeta11,tempg1) + np.dot(self.zeta12,tempg2) )+ np.dot( np.dot( tempg2 , self.B) , np.dot(self.zeta22,tempg2) + np.dot(self.zeta21,tempg1 ) )

        if mode==2: #cartesiana
            self.G(x,y,En,delta,m,pf,2)
            tempg1=self.G0
            self.G(x-x_atom,y-y_atom,En,delta,m,pf,2)
            tempg2=self.G0
            self.definitions(En,x_atom,y_atom,J1,J2,alpha,delta,m,pf,2)
            self.deltaG=np.dot(np.dot(tempg1,self.A ),np.dot(self.zeta11,tempg1) + np.dot(self.zeta12,tempg2) )+ np.dot( np.dot( tempg2 , self.B) , np.dot(self.zeta22,tempg2) + np.dot(self.zeta21,tempg1 ) )

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider
import matplotlib.gridspec as gs
import pickle
import scipy.signal as scisig

# class to run simulation and plot data using the green functions package
class simulation(Green):

    def __init__(self) -> None:
        pass

    #Explore the J parameters to match the YSR energy in the system.
    #J1=J2 for the two atoms, 1 atom at infinite distance
    def Jfinder(self,Jin=-1,Jfin=0,Jpx=200,alpha=0,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,Epx=200):
        self.J1=np.linspace(Jin,Jfin,Jpx)
        J2=np.linspace(Jin,Jfin,Jpx)
        self.En=np.linspace(0,delta,Epx) #energy
        ww = []
        v1 = []

        t0 = time.time()
        count = 0
        for i in self.J1:
            for k in self.En:
                self.dG(0,0,k+np.complex(0,dynes),1000000,100000,self.J1[count],J2[count],alpha,delta,m,pf,1)
                a=self.deltaG
                self.G(0,0,k+np.complex(0,dynes),delta,m,pf,1)
                b=self.G0
                ww.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
            count += 1
            v1.append(ww)
            ww = []
        self.map = np.array(v1)
    
    def splitMAPcalc(self,J1=-0.0296,J2=-0.0296,alpha=0,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,Epx=50,xy=(-20,20),xypx=50):
        #initialize variables
        self.En =np.linspace(0,delta,Epx)
        self.x=np.linspace(xy[0],xy[1],xypx)
        self.y=np.linspace(xy[0],xy[1],xypx)
        t0 = time.time()
        v1 = []
        v2 = []

        #simulation
        count = 0
        for i in self.x:
            self.timing(count,t0)
            for j in self.y:
                ww = []
                for k in self.En:
                    self.dG(0,0,k+np.complex(0,dynes),i,j,J1,J2,alpha,delta,m,pf,1)
                    a=self.deltaG
                    self.G(0,0,k+np.complex(0,dynes),delta,m,pf,1)
                    b=self.G0
                    ww.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
                v1.append(ww)
            v2.append(v1)
            v1 = []
            count += 1
        self.v2 = np.array(v2)
        #calculate the splitting
        self.map = self.splitCalc(mode='splitting')
        return
    #still miss implementation of split calculation with spherical FS
    def splitMAPcalcCirc(self,J1=-0.0296,J2=-0.0296,alpha=0,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,Epx=50,xy=(-20,20),xypx=50):
        #initialize variables
        self.En =np.linspace(0,delta,Epx)
        self.x=np.linspace(xy[0],xy[1],xypx)
        self.y=np.linspace(xy[0],xy[1],xypx)
        t0 = time.time()
        v1 = []
        v2 = []

        #simulation
        for i in self.x:
            if i == self.x[1]:
                #to predict job time
                t1 = time.time()
                cycle = np.round((t1-t0)/60,3)
                total = np.round(cycle*len(self.x),3)
                print('1 cycle:',cycle,'minutes --- Total time:',total,'minutes')
            for j in self.y:
                ww = []
                for k in self.En:
                    self.dG(0,0,k+np.complex(0,dynes),i,j,J1,J2,alpha,delta,m,pf,1)
                    a=self.deltaG
                    self.G(0,0,k+np.complex(0,dynes),delta,m,pf,1)
                    b=self.G0
                    ww.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
                v1.append(ww)
            v2.append(v1)
            v1 = []
        self.v2 = np.array(v2)
        #calculate the splitting
        self.map = self.splitCalc(mode='splitting')
        return

    def timing(self,i,t0):
        if i == 1:
            #to predict job time
            t1 = time.time()
            cycle = (t1-t0)/60
            total = np.round(cycle*self.cycles,3)
            if total <1:
                print('1 cycle:',np.round(cycle,3),'minutes --- Total time:',np.round(total*60,2),'seconds')
            else:
                print('1 cycle:',np.round(cycle,3),'minutes --- Total time:',total,'minutes')


    def splitLScalc(self,J1=-0.0296,J2=-0.0296,alpha=0,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,Epx=50,E0=0,direction='100',x=(-10000,10000),xypx=500):
        self.En = np.linspace(E0,delta,Epx)
        self.x=np.linspace(x[0],x[1],xypx)
        self.y=np.linspace(x[0],x[1],xypx)
        t0 = time.time()
        self.cycles = len(self.x)
        self.direction = direction
        if direction == '100':
            ww = []
            v1 = []
            for i in range(0,len(self.x)):
                self.timing(i,t0)
                for k in self.En:
                    self.dG(0,0,k+np.complex(0,dynes),self.x[i],0,J1,J2,alpha,delta,m,pf,1)
                    a=self.deltaG
                    self.G(0,0,k+np.complex(0,dynes),delta,m,pf,1)
                    b=self.G0
                    ww.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
                v1.append(ww)
                ww = []
            self.map = np.array(v1)
        if direction == '110':
            ww = []
            v1 = []
            count=0
            for i in range(0,len(self.x)):
                self.timing(i,t0)
                for k in self.En:
                    self.dG(0,0,k+np.complex(0,dynes),self.x[i],self.y[i],J1,J2,alpha,delta,m,pf,1)
                    a=self.deltaG
                    self.G(0,0,k+np.complex(0,dynes),delta,m,pf,1)
                    b=self.G0
                    ww.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
                v1.append(ww)
                ww = []
                count+=1
            print(count)
            self.map = np.array(v1)
        if direction == '120':
            ww = []
            v1 = []
            count=0
            for i in range(0,len(self.x)):
                self.timing(i,t0)
                for k in self.En:
                    self.dG(0,0,k+np.complex(0,dynes),self.x[i]*np.sqrt(3)/2,self.y[i]/2,J1,J2,alpha,delta,m,pf,1)
                    a=self.deltaG
                    self.G(0,0,k+np.complex(0,dynes),delta,m,pf,1)
                    b=self.G0
                    ww.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
                v1.append(ww)
                ww = []
                count+=1
            print(count)
            self.map = np.array(v1)
    
    def angle_dep(self,J1=-0.0296,J2=-0.0296,alpha=(0,np.pi),alphapx=200,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,Epx=50,E0=0,x=0,y=0):
        self.En = np.linspace(E0,delta,Epx)
        self.x=x
        self.y=y
        self.alphas=np.linspace(alpha[0],alpha[-1],num=alphapx)
        print(len(self.alphas))
        t0 = time.time()
        ww = []
        v1 = []
        self.cycles = len(self.alphas)
        for i in range(0,len(self.alphas)):
            self.timing(i,t0)
            for k in self.En:
                self.dG(0,0,k+np.complex(0,dynes),self.x,self.y,J1,J2,self.alphas[i],delta,m,pf,1)
                a=self.deltaG
                self.G(0,0,k+np.complex(0,dynes),delta,m,pf,1)
                b=self.G0
                ww.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
            v1.append(ww)
            ww = []
        self.map = np.array(v1)

    # to calculate the G0 of the system with SQUARED fermi surface
    def G0MAPcalcSquare(self,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,xy=(-20,20),xypx=50):
        t0 = time.time()
        self.x=np.linspace(xy[0],xy[1],xypx)
        self.y=np.linspace(xy[0],xy[1],xypx)
        v1 = []
        v2 = []
        for i in self.x:
            if i == self.x[1]:
                t1 = time.time()
                cycle = t1-t0
                total = cycle*len(self.x)
                print('1 cycle:',cycle,'allcycle:',total)
            for j in self.y:
                self.G(i,j,2*delta+np.complex(0,dynes),delta,m,pf,2)
                b=self.G0
                v1.append(-np.imag(+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
            v2.append(v1)
            v1 = []
        self.map = np.array(v2)
        return

    #to calculate the G0 of the system with CIRCULAR fermi surface

    def G0MAPcalcCircular(self,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,x=(0,100),xypx=100):
        self.x=np.linspace(x[0],x[1],xypx)
        self.phi=np.linspace(0,2*np.pi,xypx)
        v1 = []
        v2 = []
        t0 = time.time()
        # only to estimate job time 
        i=0
        for i in self.x[0:2]:
            if i == self.x[1]:
                t1 = time.time()
                cycle = t1-t0
                total = cycle*len(self.x)
                print('1 cycle:',cycle,'allcycle:',total)
            for j in self.phi:
                self.G(i,j,2*delta+np.complex(0,dynes),delta,m,pf,0)
                b=self.G0
                v1.append(-np.imag(+b[0][0]+b[1][1]+b[2][2]+b[3][3]))

        ##########real calculation##############
        v1 = []
        v2 = []
        for i in self.x:
            for j in self.phi:
                self.G(i,j,4*delta+np.complex(0,dynes),delta,m,pf,0)
                b=self.G0
                v1.append(-np.imag(+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
            v2.append(v1)
            v1 = []
        self.map = np.array(v2)
        self.map = self.map/np.max(self.map)

    #calculate the G of an isolated YSR

    def GYSRMapCalcSquare(self,J1=-0.0296,J2=-0.0296,alpha=0,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,Epx=50,xy=(-20,20),xypx=50,YSRenergy=0.000020947026906146):
        #initialize variables
        self.En =np.linspace(0,delta,Epx)
        self.x=np.linspace(xy[0],xy[1],xypx)
        self.y=np.linspace(xy[0],xy[1],xypx)
        self.cycles = len(self.x)
        t0 = time.time()
        v1 = []
        v2 = []
        count = 0
        for i in self.x:
            self.timing(count,t0)
            for j in self.y:
                self.G(i,j,YSRenergy+np.complex(0,dynes),delta,m,pf,2)
                self.dG(i,j,YSRenergy+np.complex(0,dynes),5,5,J1,0,alpha,delta,m,pf,2)
                a=self.deltaG
                v1.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]))
            v2.append(v1)
            v1 = []
            count += 1
        self.map = np.array(v2)
        self.map = self.map/np.max(self.map)
        return

    def GYSRLSCalcSquare(self,J1=-0.0296,J2=-0.0296,alpha=0,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-6,Epx=400,x=(0,457),xypx=500,YSRenergy=0.000020947026906146):
        #initialize variables
        self.En =np.linspace(0,delta,Epx)
        self.x=np.linspace(x[0],x[1],xypx)
        self.cycles = len(self.x)
        t0 = time.time()
        v1 = []
        count = 0
        for i in self.x:
            self.timing(count,t0)
            self.G(i,0,YSRenergy+np.complex(0,dynes),delta,m,pf,1)
            self.dG(i,0,YSRenergy+np.complex(0,dynes),1e10,1e10,J1,J2,alpha,delta,m,pf,1)
            a=self.deltaG
            v1.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]))
            count += 1
        self.cut = np.array(v1)
        self.cut = self.cut/np.max(self.cut)
        return

    def GYSRMapCalcCirc(self,J1=-0.0296,J2=-0.0296,alpha=0,delta=0.0000287,m=20.956,pf=0.274,dynes=5e-7,Epx=50,x=(-20,20),xypx=50,YSRenergy=0.000020947026906146):
        #initialize variables
        self.En =np.linspace(0,delta,Epx)
        self.x=np.linspace(x[0],x[1],xypx)
        self.phi=np.linspace(0,2*np.pi,xypx)
        t0 = time.time()
        v1 = []
        v2 = []
        count = 0
        for i in self.x:
            self.timing(count,t0)
            for j in self.phi:
                self.G(i,j,YSRenergy+np.complex(0,dynes),delta,m,pf,0)
                self.dG(i,j,YSRenergy+np.complex(0,dynes),5e3,5e3,J1,J2,alpha,delta,m,pf,0)
                a=self.deltaG
                v1.append(-np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]))
            v2.append(v1)
            v1 = []
            count += 1
        self.map = np.array(v2)
        self.map = self.map/np.max(self.map)
        return


    def loadMap(self,fname,xy=(-456,456)):
        self.map = np.loadtxt(fname)
        self.x= xy
        self.y= xy


    #function to plot map. Type= 
    ## general: a general 2D plot
    ## splitMap: a 2D splitting map with xy coordinate as extent
    ## linescan: a linescan (or cut of splitMap) Where
    ## Jscan: a plot varying J to find the YSR energy
    def map2Dplot(self,A='own',type='xyMap',xyunits='nm',deltaunits = 'Ha',slider='off'):
        if A == 'own': #load the map attribute of the class, so if A=!'own' can be used to plot external images
            A = self.map
        self.figure = plt.figure(figsize = (5,5))
        self.figure.subplots_adjust(bottom=0.3)
        self.axMap = self.figure.add_subplot(111)

        if type == 'xyMap':
            extent = [self.x[0],self.x[-1],self.y[0],self.y[-1]]
            aspect = None
            if xyunits == 'nm':
                a0 = const.physical_constants['atomic unit of length'][0]*1e9
                extent = [self.x[0]*a0,self.x[-1]*a0,self.y[0]*a0,self.y[-1]*a0]

        if type == 'linescan':
            A = np.flipud(A)
            extent = [self.En[0],self.En[-1],self.x[0],self.x[-1]]
            aspect = 'auto'
            if xyunits == 'nm':
                a0 = const.physical_constants['atomic unit of length'][0]*1e9
                if self.direction == '110':
                    extent = [self.En[0],self.En[-1],self.x[0]*a0*np.sqrt(2),self.x[-1]*a0*np.sqrt(2)]
                else:
                    extent = [self.En[0],self.En[-1],self.x[0]*a0,self.x[-1]*a0]
                # extent = None
                aspect = abs((extent[1]-extent[0])/(extent[3]-extent[2]))

            if deltaunits == 'delta':
                if self.direction == '110':
                    extent = [self.En[0]//self.En[-1],self.En[-1]/self.En[-1],self.x[0]*a0*np.sqrt(2),self.x[-1]*a0*np.sqrt(2)]
                else:
                    extent = [self.En[0]//self.En[-1],self.En[-1]/self.En[-1],self.x[0]*a0,self.x[-1]*a0]
                
                aspect = abs((extent[1]-extent[0])/(extent[3]-extent[2]))

        if type == 'Jscan':
            extent = [self.En[0],self.En[-1],self.J1[-1],self.J1[0]]
            aspect = 'auto'
            self.axMap.axvline(x=0.0000209470)
        
        if type == 'alphascan':
            extent = [self.alphas[0]/np.pi,self.alphas[-1]/np.pi,self.En[0],self.En[-1]]
            aspect = 'auto'
            A = np.fliplr(np.rot90(self.map))

        self.im1 = self.axMap.imshow(A/np.max(A),extent=extent,aspect=aspect,interpolation='nearest',cmap='Blues')
        if slider == 'on':

            self.axmin = self.figure.add_axes([0.15, 0.1, 0.65, 0.03])
            self.axmax = self.figure.add_axes([0.15, 0.15, 0.65, 0.03])
            self.smin = Slider(self.axmin, 'Min', np.min(A)/2 ,np.max(A)*2, valinit =np.min(A))
            self.smax = Slider(self.axmax, 'Max', np.min(A)/2 ,np.max(A)*2, valinit =np.max(A))
            self.smin.on_changed(self.update)
            self.smax.on_changed(self.update)
        if type == 'xymap':
            divider = make_axes_locatable(self.axMap) #to properly position colorbar
            cax = divider.append_axes("right", size="5%", pad=0.05)
            self.cbar = self.figure.colorbar(self.im1,cax=cax)
        
        if type == 'linescan':
            cax = plt.axes([0.85, 0.3, 0.035, 0.581])
            self.cbar = self.figure.colorbar(self.im1,cax=cax)

    #polar plot of 2D map dependent on r and phi
    def mapPolarPlot(self,A):
        self.figure,self.axMap = plt.subplots(dpi=120,subplot_kw=dict(projection='polar'))
        self.figure.subplots_adjust(bottom=0.3)
        self.im1 = self.axMap.contourf(self.phi,self.x,A,1000,cmap='Blues',interpolation='gauss',antialiasing='on')
        self.axmin = self.figure.add_axes([0.15, 0.1, 0.65, 0.03])
        self.axmax = self.figure.add_axes([0.15, 0.15, 0.65, 0.03])
        self.smin = Slider(self.axmin, 'Min', np.min(A)/2 ,np.max(A)*2, valinit =np.min(A))
        self.smax = Slider(self.axmax, 'Max', np.min(A)/2 ,np.max(A)*2, valinit =np.max(A))
        self.smin.on_changed(self.update)
        self.smax.on_changed(self.update)
        self.axMap.set_yticklabels([])
        self.axMap.set_xticklabels([])
        self.axMap.yaxis.grid(False)
        self.axMap.xaxis.grid(False)
        # divider = make_axes_locatable(self.axMap) #to properly position colorbar
        # cax = divider.append_axes("right", size="5%", pad=0.05)
        # self.figure.colorbar(self.im1,cax=cax)
        pass

    def splitCalc(self,mode='splitting'): #choose between peak separation and monitoring only 1 peak
        self.EsplitMap = np.zeros([self.v2.shape[0],self.v2.shape[0]])
        self.mode = mode
        for i in range(0,len(self.v2)):
            for j in range(0,len(self.v2)):
                self.EsplitMap[i,j] = self.Esplitfinder(self.v2[i,j,:],mode)
        #normalization
        self.EsplitMap = self.EsplitMap/np.max(self.EsplitMap)
        return self.EsplitMap

    def splitSpec(self,mode='splitting'):
        self.spec = []
        for i in range(0,len(self.map)):
            self.spec.append(self.Esplitfinder(self.map[i],mode))
        return self.spec

    def asitesPlot(self,d=1): #to add bi2pd atomic coordinates
        c1 = plt.Circle((0.672,0.336),d,color='C3')
        c2 = plt.Circle((0.672,0.672),d,color='C2')
        c3 = plt.Circle((0,0.672),d,color='C1')

        self.axMap.add_patch(c1)
        self.axMap.add_patch(c2)
        self.axMap.add_patch(c3)

    def map2Dplotgrid(self,extent=None,aspect='auto',cmin=None,cmax=None):
        gsize = np.int(np.sqrt(len(self.fnames)))
        grid = gs.GridSpec(gsize+1,gsize+1,wspace=0.3)
        self.figure = plt.figure(figsize = (10,10))
        count = 0
        for fname in self.fnames:
            if count == len(self.fnames)-1: #nappa debug for last map
                self.v2 = self.load_obj(fname)
                self.EsplitMap = np.zeros([self.v2.shape[0],self.v2.shape[0]])
                self.axMap = self.figure.add_subplot(grid[count])
                self.axMap.tick_params(axis='y',direction='in')
                self.axMap.tick_params(axis='x',direction='in')
                self.im1 = self.axMap.imshow(self.EsplitMap,extent=extent,aspect=aspect,cmap='Blues',interpolation='None',vmin=cmin,vmax=cmax)
            else:
                self.v2 = self.load_obj(fname)
                self.EsplitMap = self.splitCalc(mode='splitting')
                self.axMap = self.figure.add_subplot(grid[count])
                self.axMap.tick_params(axis='y',direction='in')
                self.axMap.tick_params(axis='x',direction='in')
                self.im1 = self.axMap.imshow(self.EsplitMap,extent=extent,aspect=aspect,cmap='Blues',interpolation='None',vmin=cmin,vmax=cmax)
            count += 1
    
    def LSplotgrid(self,x): #to plot calculated linescans x is the extent of the linescan, 
        gsize = np.int(np.sqrt(len(self.fnames)))
        grid = gs.GridSpec(gsize+1,gsize+1)
        self.figure = plt.figure(figsize = (10,10))
        count = 0
        v1 = self.load_obj(self.fnames[0])
        En = np.linspace(0,0.0000287,v1.shape[1])
        count = 0
        for fname in self.fnames:
            v1 = self.load_obj(fname)
            self.axMap = self.figure.add_subplot(grid[count])
            self.im1 = self.axMap.imshow(v1,extent=[En[0],En[-1],-x,x],aspect='auto',cmap='Blues')
            count += 1

    def LSplotgridfromMap(self,x,idxcut): #to plot linescans cut from 2D maps x is the extent of the linescan
        gsize = np.int(np.sqrt(len(self.fnames)))
        grid = gs.GridSpec(gsize+1,gsize+1,wspace=0.3)
        self.figure = plt.figure(figsize = (10,10))
        count = 0
        v1 = self.load_obj(self.fnames[0])
        En = np.linspace(0,0.0000287,v1.shape[1])
        count = 0
        for fname in self.fnames:
            v2 = self.load_obj(fname)
            v1 = v2[:][idxcut][:]
            self.axMap = self.figure.add_subplot(grid[count])
            self.axMap.tick_params(axis='y',direction='in')
            self.axMap.tick_params(axis='x',direction='in')
            self.im1 = self.axMap.imshow(v1,extent=[En[0]/En[-1],En[-1]/En[-1],-x,x],aspect='auto',cmap='Blues')
            count += 1

    def update(self, val): #for the color scale sliders
        self.im1.set_clim([self.smin.val,self.smax.val])
        self.figure.canvas.draw()

    def Esplitfinder(self,spectra,mode):#given a spectra finds the splitting
        p_idx = scisig.find_peaks(spectra)
        Esplit = 0

        if len(p_idx[0]) == 1 and self.En[p_idx[0][0]] < 0.000018: #check if there is only 1 peak and that the peak is at least below the initial YSR energy
            Esplit = -(self.En[p_idx[0][0]]-2.855e-5)
        elif len(p_idx[0]) == 2: #if there are two peaks decide which method to use
            if mode == 'splitting': #difference between energies
                Esplit = -(self.En[p_idx[0][0]]-self.En[p_idx[0][1]])
            elif mode == 'lowerPeak': #energy of the lower peak
                Esplit = self.En[p_idx[0][0]]
        # elif 80<i<115 and 80<j<115:#to avoid 0 spot in the middle
            # Esplit = -(self.En[p_idx[0][0]]-0.0000287)
        else:
            pass
        return Esplit

    #from a calculated map (v2) extract a linecut of spectras
    def LScut(self,direction='100'):
        if direction == '110':
            self.mapCut = np.rot90(self.v2.diagonal(axis1=1,axis2=0))
        self.map2Dplot(self.mapCut,type='linescan')
        pass

    def save_obj(self,obj, name ):
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 
    def load_obj(self, name ):
        with open(name, 'rb') as f:
            return pickle.load(f)

#convert the map calculated with angle_dep function from 2D plot to array of points looking for the maximas (YSR)

    def mapTospec(self):
        count = 0
        a2line_alpha = []
        a2line_E = []
        for spec in self.map: 
            idx = scisig.find_peaks(spec)
            if len(idx[0]) == 1:
                a2line_alpha.append(self.alphas[count])
                a2line_E.append(self.En[idx[0]])
            if len(idx[0]) == 2:
                a2line_alpha.append(self.alphas[count])
                a2line_E.append(self.En[idx[0][0]])
                a2line_alpha.append(self.alphas[count])
                a2line_E.append(self.En[idx[0][1]])
            count += 1
        self.spec = (np.array(a2line_alpha),np.array(a2line_E))

import sys
sys.path.append('C:/Users/strivini/OneDrive - Asociacion Cic Nanogune/PhD LOWTSTM/Python/Modules_lastversion')
import nanonis
import functions
from lmfit import Model

class fitspec(Green):
    def __init__(self,Epx,dimer,mode='isolated',T=1.5) -> None:
        ##### PARAMETERS INITIALIZATION #####
        self.Eh = const.physical_constants['atomic unit of energy'][0]
        self.deltas = 2.87e-5
        self.deltat = 2.36e-5
        self.En = np.linspace(-8*self.deltas,8*self.deltas,Epx)
        self.Vn = np.linspace(-4*self.deltas,4*self.deltas,Epx)
        self.T = T
        if dimer == '100':
            self.x1 = 0
            self.x2 = 12.7
        elif dimer == '110':
            self.x1 = 12.7
            self.x2 = 12.7
        elif dimer == '120':
            self.x1 = 6.35
            self.x2 = 12.7
        
        if mode == 'dimer':
            self.J1 = -0.0296
            self.J2 = -0.0296
        elif mode == 'isolated':
            self.J1 = -0.0236
            self.J2 = 0
        elif mode == 'BCS':
            self.J1 = -0.0296
            self.J2 = 0

    def load(self,filename,offset=0):
        self.spectra = nanonis.biasSpectroscopy()
        self.spectra.load(filename)
        self.spectra.biasOffset(offset)
        self.spectra.conductance = self.spectra.conductance/self.spectra.conductance[0]
        #convert to atomic units
        self.spectra.bias = self.spectra.bias*const.e/self.Eh


    def dynesdos(self, E, Gamma): #dynes function
        dos = np.real((E-1j*Gamma)/np.sqrt((E-1j*Gamma)**2-self.deltat**2))
        return np.abs(dos)    

    def fdd(self, E,mu, T): #fermi Dirac function
        if T == 0:
            f = np.heaviside((E-mu), 1)
        else:
            f = 1/(1+np.exp((E-mu)/(const.k*T/self.Eh)))
        return f

    def YSRdos(self,Gamma,alpha,m=20.956,pf=0.274,c=1):
        ww = []
        for V in self.En*c:
            self.dG(0,0,V+np.complex(0,Gamma),self.x1,self.x2,self.J1,self.J2,alpha,self.deltas,m,pf,1)
            a=self.deltaG
            self.G(0,0,V+np.complex(0,Gamma),self.deltas,m,pf,1)
            b=self.G0
            ww.append(-np.imag(a[0][0]+a[1][1]+b[0][0]+b[1][1]))#a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3]))
        return ww

    #convolution with toepliz matrix (Fast)
    def dynesConvT(self,bias,Gamma,alpha):
        #store the parameters as fit start
        self.Gamma = Gamma
        self.alpha = alpha

        A,B = np.meshgrid(self.Vn,self.En)
        toep = A+B
        print(np.size(toep))
        #generate linear dos
        sample = self.YSRdos(Gamma,alpha)
        fermi = self.fdd(self.En,0,self.T)
        # generate toepliz 
        tipT = self.dynesdos(toep,Gamma)
        fermiT = self.fdd(toep,0,self.T)
        #convolution with toepliz matrix
        curr = np.dot(np.multiply(sample,fermi),tipT)-np.dot(sample,np.multiply(tipT,fermiT))
        #normalization
        didv = np.gradient(np.array(curr))
        didv = didv/didv[0]
        return didv

    def fitModel(self):
        model = Model(self.dynesConvT)
        params = model.make_params()
        params['Gamma'].set(self.Gamma,vary=False)
        params['alpha'].set(self.alpha,vary=False)
        
        #perform fit
        self.fit_res  = model.fit(self.spectra.conductance,bias=self.spectra.bias)
        self.fit_res_eval = self.fit_res.eval(x=self.bias)

    def showResults(self):
        fig,self.ax = plt.subplots(1)
        self.ax.plot(self.bias,self.conductance)
        self.ax.plot(self.bias,self.fit_res.eval(x=self.bias))

    def model_init():
        pass
    
    def auto_fit():
        pass

    def manual_fit():
        pass


    # OLD FUNCTIONS
        # genera toepliz matrix for YSR dos
    def YSRdosT(self,toep,dynes,x,y,J1=-0.0296,J2=-0.0296,alpha=0,deltas=0.0000287,m=20.956,pf=0.274,c=1):
        YSRdosT = np.zeros(toep.shape)
        for i in range(0,toep.shape[0]):
            for j in range(0,toep.shape[1]):
                self.dG(0,0,toep[i,j]+np.complex(0,dynes),y,x,J1,J2,alpha,deltas,m,pf,1)
                a=self.deltaG
                self.G(0,0,toep[i,j]+np.complex(0,dynes),deltas,m,pf,1)
                b=self.G0
                YSRdosT[i,j] = -np.imag(a[0][0]+a[1][1]+a[2][2]+a[3][3]+b[0][0]+b[1][1]+b[2][2]+b[3][3])
        return YSRdosT

    # convolution with np.trapz module (Slow)
    def dynesConv(self,bias,A,deltat=0.0000287,dynest=5e-7,T=1.5):
        self.V = np.linspace(-4*self.delta,4*self.delta,200)
        curr = []
        for V in self.V:
            currp = np.trapz(self.YSRdos(5e-7,2,2)*self.dynesdos(self.En+V,dynest,deltat)*
            (self.fdd(self.En,0,T)-self.fdd(self.En+V,0,T)), x=self.En)
            curr.append(currp)
        return -A*np.gradient(np.array(curr))
