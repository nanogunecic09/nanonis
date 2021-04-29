
from modules import nanonis
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

#added: now the linescans cuts save properly in order
#       new class for the lineprofile cuts, given a folder with cuts it plot them

class lineProfile():

    def __init__(self, vmin=0, vmax=4,influence='off',range=0,plugins='on'): # vmin/vmax colourscale, cut=True enables vertical cuts
        self.vmax = vmax
        self.vmin = vmin
        self.range = range
        self.influence = influence
        self.figure = plt.figure(figsize = (5,5))

        if plugins=='on':
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
            #self.colormap = 'YlGnBu_r'
        if plugins=='off':
            self.axMap = self.figure.add_subplot(111)
        self.figure.show()

    def draw(self):
        self.im1 = self.axMap.imshow(np.fliplr(self.linescan.conductance), aspect='auto', extent=[min(self.linescan.bias), max(self.linescan.bias), min(self.linescan.distance), max(self.linescan.distance)],interpolation=None, vmin=self.vmin, vmax=self.vmax)
        #self.figure.colorbar(self.im1) #buggy colorbar

    def mapLoad(self,filenames):
        if len(filenames)>1:
            self.linescan = nanonis.linescan()
            self.linescan.load(filenames)
            self.axMap.set_title(self.linescan.name[0]+' - '+self.linescan.name[-1], fontweight='bold')
            self.cmin = self.linescan.conductance.min()
            self.cmax = self.linescan.conductance.max()
            self.axMap.set_ylabel("Distance (nm)")
            self.axMap.set_xlabel("Bias (mV)")
            plt.subplots_adjust(hspace=0.35)
            self.draw()
        if len(filenames) == 1: #checks if we load a 3ds files instead of ascii files
            self.linescan = nanonis.linescan3ds()
            self.linescan.load(filenames[0])
            self.axMap.set_title(self.linescan.name)
            self.cmin = self.linescan.conductance.min()
            self.cmax = self.linescan.conductance.max()
            self.axMap.set_ylabel("Distance (nm)")
            self.axMap.set_xlabel("Bias (mV)")
            plt.subplots_adjust(hspace=0.35)
            self.draw()


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
        if event.inaxes == self.axMap:
            if event.dblclick:
                self.spectraIndex = event.y 
            elif self.influence == 'off':
                self.energyCut = event.xdata
                self.cutPlot(self.energyCut)
            else:
                self.energyCut = event.xdata
                self.cutPlotRange(self.energyCut)
    
    def cutPlot(self, energy):
        bias = np.linspace(self.linescan.bias[0],self.linescan.bias[-1], len(self.linescan.bias))
        id = (abs(bias-energy)).argmin()
        self.axMap.plot([self.linescan.bias[id],self.linescan.bias[id]],[self.linescan.distance[0],self.linescan.distance[-1]])
        self.axCut.plot(self.linescan.distance, self.linescan.conductance[:,id], label=str(round(self.linescan.bias[id],2)))
        self.axCut.legend()
        self.saveCSV(self.linescan.distance, self.linescan.conductance[:,id])
        self.figure.canvas.draw_idle()
    
    def cutPlotRange(self, energy):
        def LSinfluence_avg(id_c,id_n,id_p): #return contuctance averaged between different cuts
            idxs = np.arange(id_n,id_p)
            LSconductance_avg = np.zeros(len(self.linescan.conductance))
            if id_n == id_p: #checks that the range is >0
                LSconductance_avg = self.linescan.conductance[:,id_c]
            else:
                for idx in idxs:
                    LSconductance_avg = LSconductance_avg + self.linescan.conductance[:,idx]
                LSconductance_avg = LSconductance_avg/len(idxs)
            return LSconductance_avg
        
        bias = np.linspace(self.linescan.bias[0], self.linescan.bias[-1],len(self.linescan.bias))
        #calculate the index based on the range given and the energy
        id_c = (abs(bias-energy)).argmin()
        id_n = (abs(bias-energy-self.range)).argmin()
        id_p = (abs(bias-energy+self.range)).argmin()
        LSinfluence_avg = LSinfluence_avg(id_c,id_n,id_p)
        if id_n == id_p:
            self.axMap.plot([self.linescan.bias[id_c],self.linescan.bias[id_c]],[self.linescan.distance[0],self.linescan.distance[-1]])
        else:
            self.axMap.fill_between([self.linescan.bias[id_n],self.linescan.bias[id_p]],self.linescan.distance[0],self.linescan.distance[-1],alpha=0.6)
        self.axCut.plot(self.linescan.distance,LSinfluence_avg,label=str(round(self.linescan.bias[id_c],2)))
        self.axCut.legend()
        self.saveCSV(self.linescan.distance, LSinfluence_avg)
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
                if processing is 'firstPoint':
                    for i in range(self.map.shape[1]):
                        self.map[i][:] = self.map[i][:]-self.map[i][0]
                if processing is 'points':
                    for i in range(self.map.shape[1]):
                        self.map[i][:] = self.map[i][:]-np.average(self.map[i][:20])
                if processing is 'average':
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
        return self.map

    def resetClick(self, event):
        self.axCut.cla()

    def fftClick(self, event):
        fft = np.fft.fft2(self.map)
        fft = fft+np.rot90(np.rot90(fft))
        fft = abs(np.fft.fftshift(fft))
        self.axCut.imshow(fft, cmap='magma_r')

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


