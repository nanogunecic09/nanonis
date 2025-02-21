from pandas.core.reshape.tile import cut
from . import nanonis
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib import rc, ticker
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from scipy import ndimage
import csv
import glob as glob
from matplotlib.widgets import Slider
import os
import pandas as pd
import colorcet as cc
# from lmfit import Model
from scipy import stats
import useful as uf
#added: now the linescans cuts save properly in order
#       new class for the lineprofile cuts, given a folder with cuts it plot them
#       new class for plotting general image with sliders


#lineProfile() is a class dependent on nanonis.linescan(). Adds the graphing features:
# if the plugins are on: sliders for vmax vmin, vertical cut with or without the influence radius
# cutmode different than line allows to click and retrieve the point number saved in self.points
# flipud='on' flips the map up-down
# noline='on' is for loading datasets that are not linescans but we still want to plot them with imshow.
# LStype change allows to load different LS formats. Can be: 'normal', the conventional set of .dat files, 'wsxm' a txt file with a LS extracted from WSXM, .3ds a linescan saved in binary.
class lineProfile():

    def __init__(self,spac=0,categorical='python',plotmode='cmap',influence='off',range=0.1e-3,plugins='on',cutMode='line',flipud='off',LStype='normal'): # vmin/vmax colourscale, cut=True enables vertical cuts
        #import the settings for the cmap plotting
        self.cutMode = cutMode
        self.LStype = LStype
        self.range = range
        self.colormap = 'YlGnBu_r'
        self.influence = influence
        self.flipud= flipud
        #import settings for cascade plotting
        self.plotmode, self.spac, self.categorical = plotmode,spac, categorical


        self.figure = plt.figure(figsize = (5,3))
        if plugins=='on': # enables the cmap changes 
            self.figure.subplots_adjust(bottom=0.3)
            grid = gs.GridSpec(2, 1, height_ratios=[2, 1])
            self.axMap = self.figure.add_subplot(grid[0])
            self.axCut = self.figure.add_subplot(grid[1])
            self.axCut.set_xlabel('Distance (nm)')
            self.axCut.set_ylabel('dI/dV (arb. units)')
            self.figure.canvas.mpl_connect('button_press_event', self.mapClick)
            self.axmin = self.figure.add_axes([0.25, 0.1, 0.65, 0.03])
            self.axmax = self.figure.add_axes([0.25, 0.15, 0.65, 0.03])
            self.smin = Slider(self.axmin, 'Min', -4, 8, valinit =0)
            self.smax = Slider(self.axmax, 'Max', -4, 8, valinit =4)
            self.smin.on_changed(self.update)
            self.smax.on_changed(self.update)
        if plugins=='off':
            self.axMap = self.figure.add_subplot(111)
        self.figure.show()

    def draw(self):
        if self.plotmode == 'cmap':
            self.im1 = self.axMap.imshow(self.linescan.conductance, aspect='auto', extent=self.extent,cmap = self.colormap, interpolation='nearest', vmin=self.vmin, vmax=self.vmax)
        if self.plotmode == 'cascade':
            if self.categorical == 'viridis':
                N = self.linescan.conductance.shape[0]
                plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,N)))
            for i in range(0,N):
                self.axMap.plot(self.linescan.bias*1e3,self.linescan.conductance[i,:]+i*self.spac)

    def mapLoad(self,filenames,distance=None): # distance is the path of the distance file for wsxm LS type
        if self.LStype == 'normal': #for linescans recorded in series of .dat files
            self.linescan = nanonis.linescan()
            self.linescan.load(filenames)
            self.axMap.set_title(self.linescan.name[0]+' - '+self.linescan.name[-1], fontweight='bold')
            self.cmin = self.linescan.conductance.min()
            self.cmax = self.linescan.conductance.max()
            self.vmin = self.linescan.conductance.min()
            self.vmax = self.linescan.conductance.max()
            self.axMap.set_ylabel("Distance (nm)")
            self.axMap.set_xlabel("Bias (mV)")
            plt.subplots_adjust(hspace=0.35)
            self.extent = [min(self.linescan.bias*1e3), max(self.linescan.bias*1e3), min(self.linescan.distance), max(self.linescan.distance)]
            if self.flipud=='on':
                self.linescan.conductance = np.flipud(self.linescan.conductance)
            self.draw()
        if self.LStype == '3ds': #checks if we load a 3ds files instead of ascii files
            self.linescan = nanonis.linescan3ds()
            self.linescan.load(filenames)
            self.axMap.set_title(self.linescan.name)
            self.vmin = self.linescan.conductance.min()
            self.vmax = self.linescan.conductance.max()
            self.cmin = self.linescan.conductance.min()
            self.cmax = self.linescan.conductance.max()
            self.axMap.set_ylabel("Distance (nm)")
            self.axMap.set_xlabel("Bias (mV)")
            plt.subplots_adjust(hspace=0.35)
            self.extent = [min(self.linescan.bias*1e3), max(self.linescan.bias*1e3), min(self.linescan.distance), max(self.linescan.distance)]
            self.draw()
        if self.LStype == 'wsxm': #for linescans extracted from wsxm, need a distance file path
            self.linescan = nanonis.linescan()
            self.linescan.conductance,self.linescan.bias,self.linescan.distance = nanonis.readGcutLS(filenames,distance)
            self.linescan.name = filenames
            self.axMap.set_title(filenames)
            self.cmin = self.linescan.conductance.min()
            self.cmax = self.linescan.conductance.max()
            self.vmin = self.linescan.conductance.min()
            self.vmax = self.linescan.conductance.max()
            self.axMap.set_ylabel("Distance (nm)")
            self.axMap.set_xlabel("Bias (mV)")
            plt.subplots_adjust(hspace=0.35)
            self.extent = [min(self.linescan.bias), max(self.linescan.bias), min(self.linescan.distance), max(self.linescan.distance)]
            self.draw()

    def mapCut(self,range=[-2e-3,2e-3]):
        idxs = [abs(self.linescan.bias-range[0]).argmin(),abs(self.linescan.bias-range[1]).argmin()]
        print(idxs)
        self.linescan.conductance = self.linescan.conductance[:,idxs[-1]:idxs[0]]
        self.linescan.bias = self.linescan.bias[idxs[-1]:idxs[0]]
        self.extent = [min(self.linescan.bias*1e3), max(self.linescan.bias*1e3), min(self.linescan.distance), max(self.linescan.distance)]
  
    def mapScale(self, min, max):
        self.cmin = min
        self.cmax = max
        self.axMap.cla()
        self.draw()

    def mapColor(self, mapCol):
        self.colormap = mapCol
        self.axMap.cla()
        self.draw()
    
    def mapGaussian(self, radius):
        self.linescan.conductance = ndimage.gaussian_filter(self.linescan.conductance, radius)
        self.axMap.cla()
        self.draw()

    def distanceOffset(self, offset):
        self.linescan.distanceOffset(offset)
        self.axMap.cla()
        self.draw()

    def normalize_offset(self, energy, offset):
        self.linescan.normalizeTo(energy)
        self.linescan.biasOffset(offset)
        self.axMap.cla()
        self.axMap.set_title(self.linescan.name[0]+' - '+self.linescan.name[-1], fontweight='bold')
        self.cmin = self.linescan.conductance.min()
        self.cmax = self.linescan.conductance.max()
        self.axMap.set_ylabel("Distance (nm)")
        self.axMap.set_xlabel("Bias (mV)")
        plt.subplots_adjust(hspace=0.35)
        self.draw()
    
    def normalize_range(self,E_range,offset):
        self.linescan.normalizeRange(E_range)
        self.linescan.biasOffset(offset)
        self.axMap.cla()
        self.axMap.set_title(self.linescan.name[0]+' - '+self.linescan.name[-1], fontweight='bold')
        self.cmin = self.linescan.conductance.min()
        self.cmax = self.linescan.conductance.max()
        self.axMap.set_ylabel("Distance (nm)")
        self.axMap.set_xlabel("Bias (mV)")
        plt.subplots_adjust(hspace=0.35)
        self.draw()

    def mapClick(self,event):
        if self.cutMode == 'point':
            if event.inaxes == self.axMap:
                if event.dblclick: #cancel the last tracked point
                    self.YSRtrace[0] = self.YSRtrace[0][:-2]
                    self.YSRtrace[1] = self.YSRtrace[1][:-2] 
                elif self.influence == 'off':
                    self.energyCut = event.xdata
                    self.distCut = event.ydata
                    self.points = [[],[]]
                    self.cutSave(self.energyCut,self.distCut)
                else:
                    self.energyCut = event.xdata
                    self.distCut = event.ydata
                    self.cutPlotRange(self.energyCut,self.distCut)
        elif self.cutMode == 'line':
            if event.inaxes == self.axMap:
                if event.dblclick:
                    self.spectraIndex = event.y 
                elif self.influence == 'off':
                    self.energyCut = event.xdata
                    self.cutPlot(self.energyCut*1e-3)
                else:
                    self.energyCut = event.xdata
                    self.cutPlotRange(self.energyCut*1e-3)
                    self.distCut = event.ydata
                    self.points = [[],[]]
                    self.cutSave(self.energyCut,self.distCut)         
    
    def cutSave(self,energy,distCut): #save the point clicked on the plot and put a dot in the graph
        id = (abs(self.linescan.bias-energy)).argmin()
        id_d = (abs(self.linescan.distance-distCut)).argmin()
        self.axMap.scatter(self.linescan.bias[id],self.linescan.distance[id_d])
        self.points[0].append(self.linescan.distance[id_d])
        self.points[1].append(self.linescan.bias[id])
        self.saveCSV(self.linescan.distance, self.linescan.conductance[:,id])
        self.figure.canvas.draw_idle()

    def cutPlot(self, energy):
        id = (abs(self.linescan.bias-energy)).argmin()
        self.axMap.plot([self.linescan.bias[id]*1e3,self.linescan.bias[id]*1e3],[self.linescan.distance[0],self.linescan.distance[-1]])
        self.axCut.plot(self.linescan.distance, self.linescan.conductance[:,id], label=str(round(self.linescan.bias[id]*1e3,2)))
        self.axCut.legend()
        self.saveCSV(self.linescan.distance, self.linescan.conductance[:,id])
        self.figure.canvas.draw_idle()
    
    def cutPlotRange(self, energy):
        def LSinfluence_avg(id_c,id_n,id_p): #return contuctance averaged between different cuts
            idxs = np.arange(id_p,id_n)
            LSconductance_avg = np.zeros(len(self.linescan.conductance))
            if id_n == id_p: #checks that the range is >0
                LSconductance_avg = self.linescan.conductance[:,id_c]
            else:
                for idx in idxs:
                    LSconductance_avg = LSconductance_avg + self.linescan.conductance[:,idx]
                LSconductance_avg = LSconductance_avg/len(idxs)
            return LSconductance_avg
        
        #calculate the index based on the range given and the energy
        id_c = (abs(self.linescan.bias-energy)).argmin()
        id_n = (abs(self.linescan.bias-energy-self.range)).argmin()
        id_p = (abs(self.linescan.bias-energy+self.range)).argmin()
        self.conductance_avg = LSinfluence_avg(id_c,id_n,id_p)
        self.axMap.fill_between([self.linescan.bias[id_n]*1e3,self.linescan.bias[id_p]*1e3],self.linescan.distance[0],self.linescan.distance[-1],alpha=0.6)
        self.axCut.plot(self.linescan.distance,self.conductance_avg,label=str(round(self.linescan.bias[id_c]*1e3,4)))
        self.axCut.legend()
        # self.saveCSV(self.linesan.distance, self.conductance_avg)
        self.figure.canvas.draw_idle()
         
    def saveCSV(self, array1,array2):
        #for incremental save
        stridx = self.linescan.name[0].find('LS')
        LSidx = self.linescan.name[0][stridx:stridx+4]
        Eidx = np.round(self.energyCut,2)
        count = 0
        for i in os.listdir():
            if 'Cut{}'.format(count) in i:
                count += 1
        matrix = np.vstack((array1,array2))
        matrix = np.transpose(matrix)
        filename_mod = "Cut{}_{}_{}.txt".format(count,LSidx,Eidx)
        with open(filename_mod, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in matrix]


    def update(self, val): #for the color scale sliders
        self.im1.set_clim([self.smin.val,self.smax.val])
        self.figure.canvas.draw()

    
    def stepfinder(self,delta,cut,change): #to correct the step caused by the rolf divider
        self.offset_steps_a = np.zeros(5)
        self.offset_steps_b = np.zeros(5)
        
        deltaIdx = self.linescan.energyFind(delta)
        cutIdx = self.linescan.energyFind(cut)
        count_a = 0
        count_b = 0
        diff = abs(self.linescan.conductance[0,deltaIdx]-self.linescan.conductance[0,0])
        print(diff)
        for i in range(1, self.linescan.conductance.shape[0]):
            if self.linescan.conductance[i,cutIdx] - self.linescan.conductance[i-1,cutIdx] > diff*change:
                self.offset_steps_a[count_a] = int(i)
                count_a +=1
        for i in range(1, self.linescan.conductance.shape[0]):
            if self.linescan.conductance[i-1,cutIdx] - self.linescan.conductance[i,cutIdx] > diff*change:
                self.offset_steps_b[count_b] = int(i)
                count_b +=1
            self.offset_steps_a = self.offset_steps_a.astype(int)
            self.offset_steps_b = self.offset_steps_b.astype(int)
        return self.offset_steps_a, self.offset_steps_b 

    def random_offset(self, offset=0, delta=0,cut=0,change=0):
        self.stepfinder(delta,cut,change)
        print(self.stepfinder(delta,cut,change))
        energy_px = self.linescan.bias[0] - self.linescan.bias[1]
        offset_px = int(offset/energy_px)
        for i in range (0,len(self.offset_steps_a)):
            if self.offset_steps_b[i] == 0:
                self.linescan.conductance[self.offset_steps_a[i]:-1,:] = np.roll(self.linescan.conductance[self.offset_steps_a[i]:-1,:], -offset_px)
                break
            self.linescan.conductance[self.offset_steps_a[i]:self.offset_steps_b[i],:] = np.roll(self.linescan.conductance[self.offset_steps_a[i]:self.offset_steps_b[i],:], -offset_px)
            print('rolled')

    def hand_normalization(self,path_values,offset):
        values = pd.read_csv(path_values,header=None)
        values = values.to_numpy()
        for i in range(len(self.linescan.name)):
            self.linescan.conductance[i][:] = self.linescan.conductance[i][:]/values[i]
        self.linescan.biasOffset(offset)
        self.axMap.cla()
        self.axMap.set_title(self.linescan.name[0]+' - '+self.linescan.name[-1], fontweight='bold')
        self.cmin = self.linescan.conductance.min()
        self.cmax = self.linescan.conductance.max()
        self.axMap.set_ylabel("Distance (nm)")
        self.axMap.set_xlabel("Bias (mV)")
        plt.subplots_adjust(hspace=0.35)
        self.draw()

    def deconvolution(self,gap=1.37e-3, temperature=1.3, dynesParameter=40e-6, energyR=8e-3, spacing=35e-6,x_min=-4E-3,x_max=4E-3,N=300, window=15,order=2,n=2000):
        self.linescan.deconvolution(gap, temperature, dynesParameter, energyR, spacing,x_min,x_max,N, window,order,n)
        self.linescan.bias = np.flip(self.linescan.bias_dec) # bias flip cause in graphix the standard is from positive to negative
        self.linescan.conductance = self.linescan.conductance_dec

class multilineprofile():
    def LSload(self, start, end ,datestamp, path):
        filenames = []
        self.nLS = end - start + 1
        self.start = start
        self.end = end
        for i in range(start, end + 1):
            if i <10:
                textchunk = 'LS0' + str(i)
            else: 
                textchunk = 'LS' + str(i)
            filenames.append(glob.glob(path + textchunk + datestamp))
        return filenames

    def axis_setup(self): # removes the axis exept the bottom ones and apply add labels
        all_axes = self.figure.get_axes()
        i = self.start
        for ax in all_axes:
            ax.annotate(str(i) + ' At', (1,1))
            ax.set_yticks([])
            i += 1
            if ax.is_last_row():
                pass
            else: 
                ax.set_xticks([])

    def plotSetup(self, ): #plot settings
        fullwith = False
        if fullwith == True:
            self.extent=[min(self.linescan.bias), max(self.linescan.bias), min(self.linescan.distance), max(self.linescan.distance)]
        else: 
            self.extent = [-self.Ecut ,self.Ecut ,min(self.linescan.distance), max(self.linescan.distance)]
        self.cmin = 0
        self.cmax = 4

    def imgslice(self, Ecut): #cut the conductance at the given energy
        self.Ecut = Ecut
        idx = (np.abs(self.linescan.bias - Ecut)).argmin()
        self.linescan.conductance = self.linescan.conductance[:,idx:300-idx]
        return self.linescan.conductance

    def multimapdraw(self, filenames,Ecut):
        self.figure = plt.figure(figsize = (10,5))
        self.grid = gs.GridSpec(4, self.nLS // 5 + 1, wspace=0.0, hspace=0.0)
        self.figure.show()
        for i in range(0,len(filenames)):
            self.linescan = nanonis.linescan()
            self.linescan.load(filenames[i])
            self.axMap = self.figure.add_subplot(self.grid[i])
            self.linescan.conductance = np.fliplr(self.linescan.conductance)
            self.linescan.conductance = self.imgslice(Ecut) #slice the array
            self.plotSetup()
            self.axMap.imshow(self.linescan.conductance, aspect='auto', extent=self.extent, vmin=self.cmin,vmax=self.cmax)
            self.axMap.set_xlabel("Bias (mV)")
        self.axis_setup()

class map():

    def __init__(self):
        self.figure = plt.figure(figsize = (7,15))
        grid = gs.GridSpec(2, 1, height_ratios=[2, 2])
        self.axMap = self.figure.add_subplot(grid[0])
        self.axCut = self.figure.add_subplot(grid[1])
        #self.figure.canvas.mpl_connect('button_press_event', self.mapClick)
        fft_button_ax = self.figure.add_axes([0.65, 0.025, 0.1, 0.04])
        self.fft_button = Button(fft_button_ax, 'FFT', hovercolor='g')
        self.fft_button.on_clicked(self.fftClick)
        reset_button_ax = self.figure.add_axes([0.8, 0.025, 0.1, 0.04])
        self.reset_button = Button(reset_button_ax, 'Reset', hovercolor='g')
        self.reset_button.on_clicked(self.resetClick)
        self.figure.show()

        
    def channPlot(self, file, channel='Z', processing='Raw', title=[], gaussian=0, colormap='pink'):
        for j in range(len(file.data.keys())):
            channs = list(file.data.keys())
            if channel in channs[j]:
                self.map = file.data[channs[j]]
                if processing == 'firstPoint':
                    for i in range(self.map.shape[1]):
                        self.map[i][:] = self.map[i][:]-self.map[i][0]
                if processing == 'points':
                    for i in range(self.map.shape[1]):
                        self.map[i][:] = self.map[i][:]-np.average(self.map[i][:20])
                if processing == 'average':
                    for i in range(self.map.shape[1]):
                        self.map[i][:] = self.map[i][:]-np.average(self.map[i][:])
                self.map = ndimage.gaussian_filter(self.map, gaussian)
                if title:
                    self.axMap.set_title('{a} {b}\n{c} meV'.format(a=file.name, b=channs[j], c=title), fontweight='bold')
                else:
                    self.axMap.set_title('{a} {b}'.format(a=file.name, b=channs[j]), fontweight='bold')
                self.map = self.map-np.min(self.map)
                self.axMap.imshow(self.map*1e9, extent=[0, file.xrange, 0, file.yrange], interpolation='mitchell', cmap=colormap)
                break

    def resetClick(self, event):
        self.axCut.cla()

    def fftClick(self, event):
        fft = np.fft.fft2(self.map)
        fft = fft+np.rot90(np.rot90(fft))
        fft = abs(np.fft.fftshift(fft))
        self.axCut.imshow(fft, cmap='magma_r',vmax=10e-8)

class lineprofileCuts():
    def __init__(self,directory,delimiter=None):
        #read a path and return the filenames of the cuts
        self.cuts_path = []
        for i in os.listdir(directory):
            if '.txt' in i:
                self.cuts_path.append(i)
        #pack the cuts in a dataframe
        dfs = []
        for i in self.cuts_path:
            df = 0
            df = pd.read_csv(directory + '/' + i,delimiter=delimiter,header=None,names=[i +'_d',i +'_I'])
            dfs.append(df)
        self.df = pd.concat(dfs,axis=1)


    def plot_cuts(self,vertOff=0,xOffset=0): #plot the cuts with a vertical offset
        self.fig, ax = plt.subplots(1)
        count = 0
        for i in range(len(self.df.columns)):
            if count==len(self.df.columns):
                break
            ax.plot(self.df[self.df.columns[count]]+xOffset, self.df[self.df.columns[count+1]]+ count*vertOff, label=self.df.columns[count])
            count += 2
            xOffset=0 #only the first spectra is going to be offsetted
        ax.set_xlabel('Distance(nm)')
        ax.set_ylabel('dI/dV')
        plt.legend()
        plt.show()
        return
    def savefig(self,name):
        self.fig.savefig('{}.pdf'.format(name))

class LScut():
    def __init__(self,side='positive',center=15,ylim=None):
        self.side = side
        self.center = center
        self.ylim = ylim

    def load(self,filename,signal):#load the data
        self.rawdata = pd.read_csv(filename,delimiter=',',header=0)
        self.data = pd.DataFrame()
        self.data[0] = self.rawdata['d']
        self.data[1] = self.rawdata[signal]
        self.shiftToZero()
        self.centerToZero()
        #define new r for plotting the fit
        self.x = np.linspace(self.data[0].iloc[0],self.data[0].iloc[-1],1000)
        if self.side == 'positive': #to avoid root of negative number
            self.x = self.x[500:]
        elif self.side == 'negative':
            self.x = self.x[:500]

    def plot1side(self):
        self.fig,self.ax = plt.subplots(1)
        self.ax.clear()
        self.ax.plot(self.data[0],self.data[1])
        self.ax.set_xlabel('Distance (nm)')
        self.ax.set_ylabel('dI/dV (mV)')
        self.ax.set_ylim(-0.2,max(self.data[1]))
        if self.side == 'positive':
            self.ax.set_xlim(0,15)
        elif self.side == 'negative':
            self.ax.set_xlim(-15,0)
        
        #sliders 1st oscillation
        self.fig.subplots_adjust(bottom=0.5)
        self.i = 0
        self.t = 0.7
        self.c1 = 23
        self.p1 = 1.5
        self.A1 = 10
        self.k1 = 1.046
        self.axi = plt.axes([0.15, 0.07, 0.65, 0.03])
        self.axc1 = plt.axes([0.15, 0.1, 0.65, 0.03])
        self.axp1 = plt.axes([0.15, 0.13, 0.65, 0.03])
        self.axA1 = plt.axes([0.15, 0.16, 0.65, 0.03])
        self.axk1 = plt.axes([0.15, 0.19, 0.65, 0.03])
        self.axt = plt.axes([0.15, 0.22, 0.65, 0.03])
        self.is1 = Slider(self.axi, 'i',  0, 30, valinit = 0)
        self.cs1 = Slider(self.axc1, 'c1',  0, 30, valinit = 23)
        self.ps1 = Slider(self.axp1, 'p1', 0, 10, valinit = 1.5)
        self.As1 = Slider(self.axA1, 'A1', 0, 20, valinit = 10)
        self.ks1 = Slider(self.axk1, 'k1', 0.5, 10, valinit = 1.046)
        self.ts = Slider(self.axt,'t', 0, 3, valinit = 0.7)
        self.is1.on_changed(self.update)
        self.cs1.on_changed(self.update)
        self.ps1.on_changed(self.update)
        self.As1.on_changed(self.update)
        self.ks1.on_changed(self.update)
        self.ts.on_changed(self.update)
        #sliders 2nd oscillation
        self.c2 = 23
        self.p2 = 1.5
        self.A2 = 10
        self.k2 = 4.48
        self.axc2 = plt.axes([0.15, 0.25, 0.65, 0.03])
        self.axp2 = plt.axes([0.15, 0.28, 0.65, 0.03])
        self.axA2 = plt.axes([0.15, 0.31, 0.65, 0.03])
        self.axk2 = plt.axes([0.15, 0.34, 0.65, 0.03])
        self.cs2 = Slider(self.axc2, 'c2',  0, 30, valinit =23)
        self.ps2 = Slider(self.axp2, 'p2', 0, 10, valinit =1.5)
        self.As2 = Slider(self.axA2, 'A2', 0, 100, valinit =10)
        self.ks2 = Slider(self.axk2, 'k2', 1, 15, valinit =4.48)
        self.cs2.on_changed(self.update)
        self.ps2.on_changed(self.update)
        self.As2.on_changed(self.update)
        self.ks2.on_changed(self.update)
        #do the plotting of the fitting functions
        if self.side == 'positive':
            y = self.oscFuncpos1(self.x,self.A1,self.A2,self.k1,self.k2,self.p1,self.p2,self.c1,self.c2,self.t,self.i)
            self.ax.plot(self.x,y)
        elif self.side == 'negative':
            y = self.oscFuncneg1(self.x,self.A1,self.A2,self.k1,self.k2,self.p1,self.p2,self.c1,self.c2,self.t,self.i)
            self.ax.plot(self.x,y)
    
    def plotFull(self):
        self.fig,self.ax = plt.subplots(1)
        self.ax.plot(self.data[0],self.data[1])
        self.ax.set_xlabel('Distance (nm)')
        self.ax.set_ylabel('dI/dV')


    def draw1D(self): #
        self.ax.clear()
        self.ax.plot(self.data[0],self.data[1])
        self.ax.set_xlabel('Distance (nm)')
        self.ax.set_ylabel('dI/dV (mV)')
        self.ax.set_ylim(-0.2,self.ylim)
        if self.side == 'positive':
            self.ax.set_xlim(0,15)
        elif self.side == 'negative':
            self.ax.set_xlim(-15,0)

    def shiftToZero(self):
        min = np.min(self.data[1])
        self.data[1] = self.data[1]-min

    def centerToZero(self):
        self.data[0] = self.data[0]-self.center

    def maskcenter(self,cutOff):
        p_idx = abs(self.data[0]-cutOff).argmin()
        n_idx = abs(self.data[0]+cutOff).argmin()

        for i in self.data[0]:
            if np.abs(i) < cutOff:
                self.data.iloc[n_idx:p_idx,1] = np.mean(self.data[1])

    def oscFuncpos1(self,x,A1,A2,k1,k2,p1,p2,c1,c2,t,i):
        oscFunc1 = (A1*(np.sin(k1*x-p1))*np.exp(-x/c1))/(np.power(x,t)*k1)
        oscFunc2 = (A2*(np.sin(k2*x-p2))*np.exp(-x/c2))/(np.power(x,t)*k2)
        return oscFunc1**2 + oscFunc2**2 + i*oscFunc1*oscFunc2
    
    def oscFuncneg1(self,x,A1,A2,k1,k2,p1,p2,c1,c2,t,i):
        oscFunc1 = (A1*(np.sin(-k1*x-p1))*np.exp(x/c1))/(np.power(-x,t)*k1)
        oscFunc2 = (A2*(np.sin(-k2*x-p2))*np.exp(x/c2))/(np.power(-x,t)*k2)
        return oscFunc1**2 + oscFunc2**2 + i*oscFunc1*oscFunc2


    def update(self, val): #for the color scale sliders
        self.c1 = self.cs1.val
        self.p1 = self.ps1.val
        self.A1 = self.As1.val
        self.k1 = self.ks1.val
        self.c2 = self.cs2.val
        self.p2 = self.ps2.val
        self.A2 = self.As2.val
        self.k2 = self.ks2.val
        self.t = self.ts.val
        self.i = self.is1.val
        self.draw1D()
        if self.side == 'positive':
            y = self.oscFuncpos1(self.x,self.A1,self.A2,self.k1,self.k2,self.p1,self.p2,self.c1,self.c2,self.t,self.i)
            self.ax.plot(self.x,y)
        elif self.side == 'negative':
            y = self.oscFuncneg1(self.x,self.A1,self.A2,self.k1,self.k2,self.p1,self.p2,self.c1,self.c2,self.t,self.i)
            self.ax.plot(self.x,y)

    def cutData(self,distCut):
        if self.side == 'positive':
            idx = abs(self.data[0]-distCut).argmin()
            self.cuttedData = self.data.iloc[idx:,:]
        if self.side == 'negative':
            idx = abs(self.data[0]-distCut).argmin()
            self.cuttedData = self.data.iloc[:idx,:]
        return

    # def autoFit(self):
    #     if self.side == 'positive':
    #         model = Model(self.oscFuncpos1)
    #         params = model.make_params()
    #         params['A1'].set(self.A1,vary=True)
    #         params['A2'].set(self.A2,vary=True)
    #         params['k1'].set(self.k1,vary=False)
    #         params['k2'].set(self.k2,vary=True)
    #         params['c1'].set(self.c1,vary=False)
    #         params['c2'].set(self.c2,vary=False)
    #         params['p1'].set(self.p1,vary=True)
    #         params['p2'].set(self.p2,vary=True)
    #         params['t'].set( self.t ,vary=False)
    #         params['i'].set( self.i ,vary=False)
    #     if self.side == 'negative':
    #         model = Model(self.oscFuncneg1)
    #         params = model.make_params()
    #         params['A1'].set(self.A1,vary=True)
    #         params['A2'].set(self.A2,vary=True)
    #         params['k1'].set(self.k1,vary=False)
    #         params['k2'].set(self.k2,vary=True)
    #         params['c1'].set(self.c1,vary=False)
    #         params['c2'].set(self.c2,vary=False)
    #         params['p1'].set(self.p1,vary=True)
    #         params['p2'].set(self.p2,vary=True)
    #         params['t'].set( self.t ,vary=False)
    #         params['i'].set( self.i ,vary=True)
    #     self.fitResult = model.fit(self.cuttedData[1],x=self.cuttedData[0],params=params)

    #     return 

    def plotfitResults(self):
        plt.figure()
        plt.plot(self.cuttedData[0],self.cuttedData[1])
        xp = np.linspace(self.cuttedData.iloc[0,0],self.cuttedData.iloc[-1,0],1000)
        comps = self.fitResult.eval(x=xp)
        plt.plot(xp,comps)

class Zapproach(nanonis.Zapproach):

    def __init__(self,vmin=0, vmax=4,influence='off',rangecut=0,plugins='on'):
        self.vmax = vmax
        self.vmin = vmin
        self.rangecut = rangecut
        self.colormap = 'YlGnBu_r'
        self.influence = influence
        self.plugins = plugins
        pass

    def draw(self):
        self.im1 = self.axMap.imshow(self.conductance, aspect='auto', extent=self.extent,cmap = self.colormap, interpolation='nearest', vmin=self.vmin, vmax=self.vmax)

    def mapload(self,fnames):
        self.load(fnames)

    
    def normalize_range(self,E_range):
        #initialize figure
        self.figure = plt.figure(figsize = (5,3))
        if self.plugins=='on':
            self.figure.subplots_adjust(bottom=0.3)
            grid = gs.GridSpec(2, 1, height_ratios=[2, 1])
            self.axMap = self.figure.add_subplot(grid[0])
            self.axCut = self.figure.add_subplot(grid[1])
            self.axCut.set_xlabel('Distance (nm)')
            self.axCut.set_ylabel('dI/dV (arb. units)')
            self.figure.canvas.mpl_connect('button_press_event', self.mapClick)
            self.axmin = self.figure.add_axes([0.25, 0.1, 0.65, 0.03])
            self.axmax = self.figure.add_axes([0.25, 0.15, 0.65, 0.03])
            self.smin = Slider(self.axmin, 'Min', -4, 8, valinit =0)
            self.smax = Slider(self.axmax, 'Max', -4, 8, valinit =4)
            self.smin.on_changed(self.update)
            self.smax.on_changed(self.update)
        if self.plugins=='off':
            self.axMap = self.figure.add_subplot(111)

        
        #load map
        self.normalizeRange(E_range)
        self.extent = [min(self.bias*1e3), max(self.bias*1e3), min(self.resistance), max(self.resistance)]

        self.axMap.set_ylabel("Resistance (Ohms)")
        self.axMap.set_xlabel("Bias (mV)")
        self.axMap.cla()
        self.draw()

    def cascade_plot(self,E_range,step,channel='didv'):
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.viridis(np.linspace(0,1,self.conductance.shape[0])))
        self.figure, self.ax = plt.subplots(1)
        
        count = 0
        if channel == 'didv':
            self.normalizeRange(E_range)
            for i in range(0,self.conductance.shape[0]):
                self.ax.plot(self.bias*1e3,self.conductance[i,:]+count*step)
                count += 1
        elif channel == 'current':
            for i in range(0,self.current.shape[0]):
                self.ax.plot(self.bias*1e3,self.current[i,:]/self.current[i,0]+count*step)
                count += 1
        plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab10(np.linspace(0,1,10)))


    def mapClick(self,event):
        if event.inaxes == self.axMap:
            self.energyCut = event.xdata
            self.cutPlotRange(self.energyCut)

    def update(self, val): #for the color scale sliders
        self.im1.set_clim([self.smin.val,self.smax.val])
        self.figure.canvas.draw()

    def cutPlotRange(self, energy):
        energy = energy*1e-3
        def LSinfluence_avg(id_c,id_n,id_p): #return contuctance averaged between different cuts
            idxs = np.arange(id_n,id_p)
            LSconductance_avg = np.zeros(len(self.conductance))
            if id_n == id_p: #checks that the range is >0
                LSconductance_avg = self.conductance[:,id_c]
            else:
                for idx in idxs:
                    LSconductance_avg = LSconductance_avg + self.conductance[:,idx]
                LSconductance_avg = LSconductance_avg/len(idxs)
            return LSconductance_avg
    
        #calculate the index based on the range given and the energy
        id_c = (abs(self.bias-energy)).argmin()
        id_n = (abs(self.bias-energy-self.rangecut)).argmin()
        id_p = (abs(self.bias-energy+self.rangecut)).argmin()
        print(id_c,id_n,id_p)
        self.conductance_avg = LSinfluence_avg(id_c,id_n,id_p)
        if id_n == id_p:
            self.axMap.plot([self.bias[id_c]*1e3,self.bias[id_c]*1e3],[self.resistance[0],self.resistance[-1]])
        else:
            self.axMap.fill_between([self.bias[id_n]*1e3,self.bias[id_p]*1e3],self.resistance[0],self.resistance[-1],alpha=0.6)
        self.axCut.plot(self.resistance,self.conductance_avg,label=str(round(self.bias[id_c]*1e3,3)))
        self.axCut.legend()
        self.saveCSV(self.resistance, self.conductance_avg)
        self.figure.canvas.draw_idle()

    def saveCSV(self, array1,array2):
        #for incremental save
        stridx = self.name[0].find('LS')
        LSidx = self.name[0][stridx:stridx+4]
        Eidx = np.round(self.energyCut,2)
        count = 0
        for i in os.listdir():
            if 'Cut{}'.format(count) in i:
                count += 1
        matrix = np.vstack((array1,array2))
        matrix = np.transpose(matrix)
        filename_mod = "Cut{}_{}_{}.txt".format(count,LSidx,Eidx)
        with open(filename_mod, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in matrix]

    #todo:influence vertical cut, plot with tools


class grid():

    def __init__(self):
        pass
    def mapload(self,filename,cmap='viridis'):
        self.cmap = cmap
        self.gridraw = nanonis.grid()
        self.gridraw.load(filename)
    
    def explorer(self,mirror=False):
        self.mirror = mirror
        self.figure = plt.figure(figsize=(6,6))
        self.axMap = self.figure.add_subplot(1,1,1)
        self.figure.subplots_adjust(bottom=0.35)
        self.ax1 = self.figure.add_axes([0.20, 0.10, 0.65, 0.03])
        self.ax2 = self.figure.add_axes([0.20, 0.15, 0.65, 0.03])
        self.ax3 = self.figure.add_axes([0.20, 0.20, 0.65, 0.03])
        # self.energyCut_slider = Slider(self.ax1,'Energy cut',self.gridraw.bias.min()*1e3,self.gridraw.bias.max()*1e3,valinit=0, valstep=(self.gridraw.bias[0]-self.gridraw.bias[1])*1e3)
        self.energyCut_slider = Slider(self.ax1,'Energy cut',self.gridraw.bias.min()*1e3,self.gridraw.bias.max()*1e3,valinit=0, valstep=(0.01))
        if self.mirror == True:
            if 'LIX 1 omega (A)' in self.gridraw.data:
                self.conductance = np.flipud(self.gridraw.data['LIX 1 omega (A)'][:,:,0])
            if 'SRX (V)' in self.gridraw.data:
                self.conductance = np.flipud(self.gridraw.data['SRX (V)'][:,:,0])
            if 'LI Demod 1 X (A)' in self.gridraw.data:
                self.conductance = np.flipud(self.gridraw.data['LI Demod 1 X (A)'][:,:,0])
        else:
            if 'LIX 1 omega (A)' in self.gridraw.data:
                self.conductance = self.gridraw.data['LIX 1 omega (A)'][:,:,0]
            if 'LIX 1 omega (A)' in self.gridraw.data:
                self.conductance = self.gridraw.data['LIX 1 omega (A)'][:,:,0]
            if 'LI Demod 1 X (A)' in self.gridraw.data:
                self.conductance = self.gridraw.data['LI Demod 1 X (A)'][:,:,0]
        self.smin_slider = Slider(self.ax2, 'Min', self.conductance.min(), self.conductance.max(), valinit =self.conductance.min())
        self.smax_slider = Slider(self.ax3, 'Max', self.conductance.min(), self.conductance.max(), valinit =self.conductance.max()*0.5)            
        self.energyCut_slider.on_changed(self.update_energy)
        self.smin_slider.on_changed(self.update_cscale)
        self.smax_slider.on_changed(self.update_cscale)
        self.im1 = self.axMap.imshow(self.conductance,extent=[0,self.gridraw.xrange,0,self.gridraw.yrange],interpolation='nearest',cmap=self.cmap,vmax=0.5,vmin=0)
        
        #energy label 
        self.label = self.axMap.text(self.gridraw.xrange/10,self.gridraw.xrange/10,'0 mV',color='white')

        #axis labels
        self.axMap.set_xlabel('x (nm)')
        self.axMap.set_ylabel('y (nm)')


    def multicut(self,energies,vmins=None,vmaxs=None): #plot multiple grid cuts given the energies
        gsize = np.int(np.sqrt(len(energies)))
        grid = gs.GridSpec(gsize+1,gsize+1,wspace=0.3)
        self.figure = plt.figure(figsize = (10,10))
        count = 0
        for energy in energies:
            self.cutIdx = (abs(self.gridraw.bias-energy*1e-3)).argmin()
            self.conductance = np.flipud(self.gridraw.data['SRX (V)'][:,:,self.cutIdx])
            self.axMap = self.figure.add_subplot(grid[count])
            if vmaxs == None:
                self.im1 = self.axMap.imshow(self.conductance,extent=[0,self.gridraw.xrange,0,self.gridraw.yrange],interpolation=None,cmap=self.cmap,vmax=np.mean(self.conductance),vmin=self.conductance.min())
            else:
                self.im1 = self.axMap.imshow(self.conductance,extent=[0,self.gridraw.xrange,0,self.gridraw.yrange],interpolation=None,cmap=self.cmap,vmax=np.mean(self.conductance),vmin=self.conductance.min())
            count += 1

    def multicutFeTPP(self,energies,labels,vmins=None,vmaxs=None): #plot multiple grid cuts given the energies
        gsize = np.int(np.sqrt(len(energies)))
        grid = gs.GridSpec(gsize+1,gsize+1,wspace=0)
        self.figure = plt.figure(figsize = (10,10))
        count = 0
        for energy in energies:
            self.cutIdx = (abs(self.gridraw.bias-energy*1e-3)).argmin()
            self.conductance = np.flipud(self.gridraw.data['SRX (V)'][:-4,4:,self.cutIdx])
            self.axMap = self.figure.add_subplot(grid[count])
            if vmaxs == None:
                self.im1 = self.axMap.imshow(self.conductance,extent=[0,3,0,3],interpolation=None,cmap=self.cmap,vmin=0,vmax=None)
            else:
                self.im1 = self.axMap.imshow(self.conductance,extent=None,interpolation=None,cmap=self.cmap,vmin=0,vmax=vmaxs[count])

            self.axMap.get_xaxis().set_visible(False)
            self.axMap.get_yaxis().set_visible(False)
            #colorbar
            if count == 1 or count == 5 or count == 9:
                axin1 = self.axMap.inset_axes(self.axMap, width='100%', height='15%', loc='upper left',bbox_to_anchor=(-0.04,0.04,0.995,1), bbox_transform=self.axMap.transAxes)
            else:
                axin1 = self.axMap.inset_axes(self.axMap, width='100%', height='15%', loc='upper left',bbox_to_anchor=(-0.04,0.04,1,1), bbox_transform=self.axMap.transAxes)
                
            axin1.get_xaxis().set_visible(False)
            axin1.get_yaxis().set_visible(False)
            axin1.tick_params(axis='both',which='both',length=0)
            self.figure.colorbar(self.im1,cax=axin1,orientation='horizontal')
            #cbar label
            axin1.text(0.05,0.40,str(np.int(self.im1.get_clim()[0])),size=12,color='w',transform = axin1.transAxes,ha='center',va="center",weight='bold')
            axin1.text(0.90,0.40,str(np.round(self.im1.get_clim()[1],1)),size=12,color='k',transform = axin1.transAxes,ha='center',va="center",weight='bold')
            axin1.text(0.90,0.40,'',size=12,color='k',transform = axin1.transAxes,ha='center',va="center",weight='bold')
            # label text
            self.axMap.text(0,0.03,labels[count],color='w',size=14,weight='bold',transform = self.axMap.transAxes)
            self.axMap.text(0.03,0.03,str(np.round(self.gridraw.bias[self.cutIdx]*1e3,2))+'mV',color='w',size=12,weight='bold',transform = self.axMap.transAxes,ha='left')
            count += 1

    def update_energy(self,val):
        self.cutIdx = (abs(self.gridraw.bias-val*1e-3)).argmin()
        if self.mirror == True:
            if 'LIX 1 omega (A)' in self.gridraw.data:
                self.conductance = np.flipud(self.gridraw.data['LIX 1 omega (A)'][:,:,self.cutIdx])
            if 'SRX (V)' in self.gridraw.data:
                self.conductance = np.flipud(self.gridraw.data['SRX (V)'][:,:,self.cutIdx])
            if 'LI Demod 1 X (A)' in self.gridraw.data:
                self.conductance = np.flipud(self.gridraw.data['LI Demod 1 X (A)'][:,:,self.cutIdx])
        else:
            if 'LIX 1 omega (A)' in self.gridraw.data:
                self.conductance = self.gridraw.data['LIX 1 omega (A)'][:,:,self.cutIdx]
            if 'LIX 1 omega (A)' in self.gridraw.data:
                self.conductance = self.gridraw.data['LIX 1 omega (A)'][:,:,self.cutIdx]
            if 'LI Demod 1 X (A)' in self.gridraw.data:
                self.conductance = self.gridraw.data['LI Demod 1 X (A)'][:,:,self.cutIdx]
        self.im1.set_data(self.conductance)
        self.im1.set_clim(np.min(self.conductance),np.max(self.conductance))
        self.label.set_text('{} mV'.format(np.round(val,2)))
        self.figure.canvas.draw()

    def update_cscale(self,val):
        self.im1.set_clim([self.smin_slider.val,self.smax_slider.val])
        self.figure.canvas.draw()


class sliderImage():
    def __init__(self) -> None:
        pass

    def load(self,map):
        self.map = map

    def plot(self,slim=None,valinit=None):
        self.figure = plt.figure(figsize = (5,5))
        self.figure.subplots_adjust(bottom=0.3)
        self.axMap = self.figure.add_subplot(111)
        self.axmin = self.figure.add_axes([0.15, 0.1, 0.65, 0.03])
        self.axmax = self.figure.add_axes([0.15, 0.15, 0.65, 0.03])

        if slim == None:
            slim = (np.min(self.map)/2,np.max(self.map)*2)
            valinit = np.max(self.map)
        self.smin = Slider(self.axmin, 'Min', slim[0] ,slim[1] , valinit =valinit)
        self.smax = Slider(self.axmax, 'Max', slim[0] ,slim[1] , valinit =valinit)
        self.smin.on_changed(self.update)
        self.smax.on_changed(self.update)

        self.im1 = self.axMap.imshow(self.map,vmin=slim[0],vmax=slim[1],aspect='auto',interpolation='nearest',cmap='Blues')



    def update(self, val): #for the color scale sliders
        self.im1.set_clim([self.smin.val,self.smax.val])
        self.figure.canvas.draw()

class ZapproachMilano():

    def __init__(self):
        pass
    def load(self,filenames):
        self.Zapproach = nanonis.Zapproach()
        self.Zapproach.load(filenames)
    
    def mapPlot(self):
        fig,ax = plt.subplots(1)
        ax.imshow(self.Zapproach.conductance,extent=[self.Zapproach.bias[-1],self.Zapproach.bias[0],0,1],aspect='auto',interpolation='nearest')
        ax.set_xlabel('Bias (mV)')

##useful plotting tools

#set the current axes in spectroscopy mode

class tri_grid():
#Falta hacer bien el MLS
    def __init__(self):
        self.type = 'Grid'
    

    def load(self,fnames,G0_set_fname,normalizeR=[4e-3,5e-3]):
        # create triangle points and load centers
        self.G0_set = uf.load_obj(G0_set_fname)
        v1 = self.G0_set['v1']
        v2 = self.G0_set['v2']
        v3 = self.G0_set['v3']
        center_x = self.G0_set['center_x'] 
        center_y = self.G0_set['center_y'] 
        width = self.G0_set['width']
        self.width = width
        height = self.G0_set['height']
        self.height = height
        spacing = self.G0_set['spacing']
        square_points = self.generate_square_grid(center_x, center_y, width, height, spacing)
        self.triangle_points = self.cut_triangle_from_square(square_points, v1, v2, v3)
        self.triangle_points = self.sort_points(self.triangle_points)
        self.x = np.arange(center_x-width/2,center_x+width/2,spacing)
        self.y = np.arange(center_y-width/2,center_y+width/2,spacing)
        n=0
        self.spectra = nanonis.biasSpectroscopy()
        self.spectra.load(fnames[0])
        self.bias = self.spectra.bias
        self.tri_grid = np.zeros((self.x.shape[0],self.y.shape[0],self.bias.shape[0]))
        for f in fnames[:-1]:
            x_idx = np.abs(self.x-self.triangle_points[n][0]).argmin()
            y_idx = np.abs(self.y-self.triangle_points[n][1]).argmin()
            self.spectra.load(f)
            # self.spectra.normalizeRange(normalizeR)
            self.tri_grid[x_idx,y_idx,:] = self.spectra.conductance
            print(x_idx,y_idx)
            n+=1

        self.x_coords = np.linspace(0,self.width,self.tri_grid.shape[0])
        self.y_coords = np.linspace(0,self.height,self.tri_grid.shape[1])
        return



    def explorer(self):
        self.figure = plt.figure(figsize=(6,6))
        self.axMap = self.figure.add_subplot(211)
        self.axSpec = self.figure.add_subplot(212)
        self.axSpec.margins(0.05)
        self.axMap.margins(0.05)
        self.figure.subplots_adjust(bottom=0.35)
        self.ax1 = self.figure.add_axes([0.20, 0.10, 0.65, 0.03])
        self.ax2 = self.figure.add_axes([0.20, 0.15, 0.65, 0.03])
        self.ax3 = self.figure.add_axes([0.20, 0.20, 0.65, 0.03])
        self.conductance = self.tri_grid[:,:,0]
        # self.energyCut_slider = Slider(self.ax1,'Energy cut',self.gridraw.bias.min()*1e3,self.gridraw.bias.max()*1e3,valinit=0, valstep=(self.gridraw.bias[0]-self.gridraw.bias[1])*1e3)
        self.energyCut_slider = Slider(self.ax1,'Energy cut',self.bias.min()*1e3,self.bias.max()*1e3,valinit=0, valstep=(0.01))
        self.smin_slider = Slider(self.ax2, 'Min', self.conductance.min(), self.conductance.max(), valinit =self.conductance.min())
        self.smax_slider = Slider(self.ax3, 'Max', self.conductance.min(), self.conductance.max(), valinit =self.conductance.max())            
        self.energyCut_slider.on_changed(self.update_energy)
        self.smin_slider.on_changed(self.update_cscale)
        self.smax_slider.on_changed(self.update_cscale)
        self.im1 = self.axMap.imshow(np.rot90(np.rot90(np.rot90(self.conductance))),extent=[0,self.width,0,self.height],interpolation='nearest',cmap='viridis',vmax=0.5,vmin=0)
        self.axvline = self.axSpec.axvline(0)
        #energy label
        self.label = self.axMap.text(self.width/10,self.height/10,'0 mV',color='white')
        #axis labels
        self.axMap.set_xlabel('x (nm)')
        self.axMap.set_ylabel('y (nm)')
        # plot of the spectra
        self.axSpec.plot(self.bias*1e3,np.mean(self.tri_grid,axis=(0,1)))
        self.figure.canvas.mpl_connect('button_press_event', self.update_spectrum_on_click)

    def update_energy(self,val):
        self.cutIdx = (abs(self.bias-val*1e-3)).argmin()
        self.conductance = self.tri_grid[:,:,self.cutIdx]
        self.im1.set_data(np.rot90(self.conductance))
        self.im1.set_clim(np.min(self.conductance),np.max(self.conductance))
        self.label.set_text('{} mV'.format(np.round(val,2)))
        self.figure.canvas.draw()
        # put index in plot
        self.axvline.remove()
        self.axvline = self.axSpec.axvline(val)

        
    def update_cscale(self,val):
        self.im1.set_clim([self.smin_slider.val,self.smax_slider.val])
        self.figure.canvas.draw()

    def update_spectrum_on_click(self, event):
        if event.inaxes == self.axMap:
            x_coord, y_coord = event.xdata, event.ydata
            x_idx = np.abs(self.x_coords - y_coord).argmin()
            y_idx = np.abs(self.y_coords - x_coord).argmin()
            selected_spectrum = self.tri_grid[x_idx,y_idx ,:]
            
            # Update the spectra plot
            self.axSpec.clear()
            self.axSpec.plot(self.bias * 1e3, selected_spectrum)
            self.axSpec.set_xlabel('Bias (mV)')
            self.axSpec.set_ylabel('Conductance')
            self.axSpec.set_title(f'Spectrum at ({x_coord}, {y_coord})')
            self.figure.canvas.draw()


    def frange(self,start, end, step):
        while start <= end:
            yield start
            start += step

    def generate_square_grid(self,center_x, center_y, width, height, spacing=1.0):
        min_x = center_x - width / 2
        max_x = center_x + width / 2
        min_y = center_y - height / 2
        max_y = center_y + height / 2
        square_points = []
        for x in self.frange(min_x, max_x, spacing):
            for y in self.frange(min_y, max_y, spacing):
                square_points.append((x, y))

        return square_points

    def is_inside_triangle(self,point, v1, v2, v3):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        b1 = sign(point, v1, v2) < 0.0
        b2 = sign(point, v2, v3) < 0.0
        b3 = sign(point, v3, v1) < 0.0

        return b1 == b2 == b3

    def cut_triangle_from_square(self,square_points, v1, v2, v3):
        triangle_points = []
        for point in square_points:
            if self.is_inside_triangle(point, v1, v2, v3):
                triangle_points.append(point)
        return triangle_points

    def sort_points(self,points):
        # Sort points first by -X (right to left), then by Y (bottom to top)
        return sorted(points, key=lambda point: (point[1], -point[0]))


class tri_grid_new():
#Falta hacer bien el MLS
    def __init__(self):
        self.type = 'Grid'
    
    def load(self,fnames,triangle_points,square_points,center_x,center_y,width,height,normalizeR=[4e-3,5e-3]):
        x = np.arange(center_x-width/2,center_x+width/2,1.5)
        y = np.arange(center_y-width/2,center_y+width/2,1.5)
        self.tri_grid = np.zeros((x.shape[0],y.shape[0],300))+0
        self.x = width
        self.y = height
        n=0
        self.spectra = nanonis.biasSpectroscopy()
        self.spectra.load(fnames[0])
        self.bias = self.spectra.bias
        for f in fnames[:-1]:
            x_idx = np.abs(x-triangle_points[n][0]).argmin()
            y_idx = np.abs(y-triangle_points[n][1]).argmin()
            self.spectra.load(f)
            self.spectra.normalizeRange(normalizeR)
            self.tri_grid[x_idx,y_idx,:] = self.spectra.conductance
            n+=1
        return
    def explorer(self):
        self.figure = plt.figure(figsize=(6,6))
        self.axMap = self.figure.add_subplot(211)
        self.axSpec = self.figure.add_subplot(212)
        self.axSpec.margins(0.05)
        self.axMap.margins(0.05)
        self.figure.subplots_adjust(bottom=0.35)
        self.ax1 = self.figure.add_axes([0.20, 0.10, 0.65, 0.03])
        self.ax2 = self.figure.add_axes([0.20, 0.15, 0.65, 0.03])
        self.ax3 = self.figure.add_axes([0.20, 0.20, 0.65, 0.03])
        self.conductance = self.tri_grid[:,:,0]
        # self.energyCut_slider = Slider(self.ax1,'Energy cut',self.gridraw.bias.min()*1e3,self.gridraw.bias.max()*1e3,valinit=0, valstep=(self.gridraw.bias[0]-self.gridraw.bias[1])*1e3)
        self.energyCut_slider = Slider(self.ax1,'Energy cut',self.bias.min()*1e3,self.bias.max()*1e3,valinit=0, valstep=(0.01))
        self.smin_slider = Slider(self.ax2, 'Min', self.conductance.min(), self.conductance.max(), valinit =self.conductance.min())
        self.smax_slider = Slider(self.ax3, 'Max', self.conductance.min(), self.conductance.max(), valinit =self.conductance.max()*0.5)            
        self.energyCut_slider.on_changed(self.update_energy)
        self.smin_slider.on_changed(self.update_cscale)
        self.smax_slider.on_changed(self.update_cscale)
        self.im1 = self.axMap.imshow(self.conductance,extent=[0,self.x,0,self.y],interpolation='nearest',cmap='viridis',vmax=0.5,vmin=0)
        #energy label 
        self.label = self.axMap.text(self.x/10,self.x/10,'0 mV',color='white')
        #axis labels
        self.axMap.set_xlabel('x (nm)')
        self.axMap.set_ylabel('y (nm)')
        # plot of the spectra
        self.axSpec.plot(self.bias*1e3,np.mean(self.tri_grid,axis=(0,1)))


    def update_energy(self,val):
        self.cutIdx = (abs(self.bias-val*1e-3)).argmin()
        self.conductance = np.flipud(self.tri_grid[:,:,self.cutIdx])
        self.im1.set_data(self.conductance)
        self.im1.set_clim(np.min(self.conductance),np.max(self.conductance))
        self.label.set_text('{} mV'.format(np.round(val,2)))
        self.figure.canvas.draw()
        # put index in plot
        self.axSpec.clear()
        self.axSpec.plot(self.bias*1e3,np.mean(self.tri_grid,axis=(0,1)))
        self.axSpec.axvline(val)
    def update_cscale(self,val):
        self.im1.set_clim([self.smin_slider.val,self.smax_slider.val])
        self.figure.canvas.draw()



class tri_grid_fix():
#Falta hacer bien el MLS
    def __init__(self):
        self.type = 'Grid'
    

    def load(self,fnames,square_points):

        self.triangle_points = self.cut_triangle_from_square(square_points, v1, v2, v3)
        self.triangle_points = self.sort_points(self.triangle_points)
        self.x = np.arange(center_x-width/2,center_x+width/2,spacing)
        self.y = np.arange(center_y-width/2,center_y+width/2,spacing)
        n=0
        self.spectra = nanonis.biasSpectroscopy()
        self.spectra.load(fnames[0])
        self.bias = self.spectra.bias
        self.tri_grid = np.zeros((self.x.shape[0],self.y.shape[0],self.bias.shape[0]))
        for f in fnames[:-1]:
            x_idx = np.abs(self.x-self.triangle_points[n][0]).argmin()
            y_idx = np.abs(self.y-self.triangle_points[n][1]).argmin()
            self.spectra.load(f)
            # self.spectra.normalizeRange(normalizeR)
            self.tri_grid[x_idx,y_idx,:] = self.spectra.conductance
            print(x_idx,y_idx)
            n+=1

        self.x_coords = np.linspace(0,self.width,self.tri_grid.shape[0])
        self.y_coords = np.linspace(0,self.height,self.tri_grid.shape[1])
        return



    def explorer(self):
        self.figure = plt.figure(figsize=(6,6))
        self.axMap = self.figure.add_subplot(211)
        self.axSpec = self.figure.add_subplot(212)
        self.axSpec.margins(0.05)
        self.axMap.margins(0.05)
        self.figure.subplots_adjust(bottom=0.35)
        self.ax1 = self.figure.add_axes([0.20, 0.10, 0.65, 0.03])
        self.ax2 = self.figure.add_axes([0.20, 0.15, 0.65, 0.03])
        self.ax3 = self.figure.add_axes([0.20, 0.20, 0.65, 0.03])
        self.conductance = self.tri_grid[:,:,0]
        # self.energyCut_slider = Slider(self.ax1,'Energy cut',self.gridraw.bias.min()*1e3,self.gridraw.bias.max()*1e3,valinit=0, valstep=(self.gridraw.bias[0]-self.gridraw.bias[1])*1e3)
        self.energyCut_slider = Slider(self.ax1,'Energy cut',self.bias.min()*1e3,self.bias.max()*1e3,valinit=0, valstep=(0.01))
        self.smin_slider = Slider(self.ax2, 'Min', self.conductance.min(), self.conductance.max(), valinit =self.conductance.min())
        self.smax_slider = Slider(self.ax3, 'Max', self.conductance.min(), self.conductance.max(), valinit =self.conductance.max())            
        self.energyCut_slider.on_changed(self.update_energy)
        self.smin_slider.on_changed(self.update_cscale)
        self.smax_slider.on_changed(self.update_cscale)
        self.im1 = self.axMap.imshow(np.rot90(np.rot90(np.rot90(self.conductance))),extent=[0,self.width,0,self.height],interpolation='nearest',cmap='viridis',vmax=0.5,vmin=0)
        self.axvline = self.axSpec.axvline(0)
        #energy label
        self.label = self.axMap.text(self.width/10,self.height/10,'0 mV',color='white')
        #axis labels
        self.axMap.set_xlabel('x (nm)')
        self.axMap.set_ylabel('y (nm)')
        # plot of the spectra
        self.axSpec.plot(self.bias*1e3,np.mean(self.tri_grid,axis=(0,1)))
        self.figure.canvas.mpl_connect('button_press_event', self.update_spectrum_on_click)

    def update_energy(self,val):
        self.cutIdx = (abs(self.bias-val*1e-3)).argmin()
        self.conductance = self.tri_grid[:,:,self.cutIdx]
        self.im1.set_data(np.rot90(self.conductance))
        self.im1.set_clim(np.min(self.conductance),np.max(self.conductance))
        self.label.set_text('{} mV'.format(np.round(val,2)))
        self.figure.canvas.draw()
        # put index in plot
        self.axvline.remove()
        self.axvline = self.axSpec.axvline(val)

        
    def update_cscale(self,val):
        self.im1.set_clim([self.smin_slider.val,self.smax_slider.val])
        self.figure.canvas.draw()

    def update_spectrum_on_click(self, event):
        if event.inaxes == self.axMap:
            x_coord, y_coord = event.xdata, event.ydata
            x_idx = np.abs(self.x_coords - y_coord).argmin()
            y_idx = np.abs(self.y_coords - x_coord).argmin()
            selected_spectrum = self.tri_grid[x_idx,y_idx ,:]
            
            # Update the spectra plot
            self.axSpec.clear()
            self.axSpec.plot(self.bias * 1e3, selected_spectrum)
            self.axSpec.set_xlabel('Bias (mV)')
            self.axSpec.set_ylabel('Conductance')
            self.axSpec.set_title(f'Spectrum at ({x_coord}, {y_coord})')
            self.figure.canvas.draw()


    def frange(self,start, end, step):
        while start <= end:
            yield start
            start += step

    def generate_square_grid(self,center_x, center_y, width, height, spacing=1.0):
        min_x = center_x - width / 2
        max_x = center_x + width / 2
        min_y = center_y - height / 2
        max_y = center_y + height / 2
        square_points = []
        for x in self.frange(min_x, max_x, spacing):
            for y in self.frange(min_y, max_y, spacing):
                square_points.append((x, y))

        return square_points

    def is_inside_triangle(self,point, v1, v2, v3):
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
        b1 = sign(point, v1, v2) < 0.0
        b2 = sign(point, v2, v3) < 0.0
        b3 = sign(point, v3, v1) < 0.0

        return b1 == b2 == b3

    def cut_triangle_from_square(self,square_points, v1, v2, v3):
        triangle_points = []
        for point in square_points:
            if self.is_inside_triangle(point, v1, v2, v3):
                triangle_points.append(point)
        return triangle_points

    def sort_points(self,points):
        # Sort points first by -X (right to left), then by Y (bottom to top)
        return sorted(points, key=lambda point: (point[1], -point[0]))
