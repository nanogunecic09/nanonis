
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
class lineProfile():

    def __init__(self, vmin, vmax,cut):
        self.vmax = vmax
        self.vmin = vmin
        self.figure = plt.figure(figsize = (5,5))
        self.figure.subplots_adjust(bottom=0.3)
        if cut==True:
            grid = gs.GridSpec(2, 1, height_ratios=[2, 1])
            self.axMap = self.figure.add_subplot(grid[0])
            self.axCut = self.figure.add_subplot(grid[1])
            self.axCut.set_xlabel('Distance (nm)')
            self.axCut.set_ylabel('dI/dV (arb. units)')
            self.figure.canvas.mpl_connect('button_press_event', self.mapClick)
            self.colormap = 'YlGnBu_r'
            self.figure.show()
        else: #without the cut
            self.axMap = self.figure.add_subplot(111)
            self.axmin = self.figure.add_axes([0.25, 0.1, 0.65, 0.03])
            self.axmax = self.figure.add_axes([0.25, 0.15, 0.65, 0.03])
            self.smin = Slider(self.axmin, 'Min', -4, 8, valinit =0)
            self.smax = Slider(self.axmax, 'Max', -4, 8, valinit =4)
            self.colormap = 'YlGnBu_r'
            self.figure.show()

    def draw(self):
        self.im1 = self.axMap.imshow(np.fliplr(self.linescan.conductance), aspect='auto', extent=[min(self.linescan.bias), max(self.linescan.bias), min(self.linescan.distance), max(self.linescan.distance)],interpolation=None, cmap=self.colormap, vmin=self.vmin, vmax=self.vmax)
        self.smin.on_changed(self.update)
        self.smax.on_changed(self.update)
        #self.figure.colorbar(self.im1) #buggy colorbar

    def mapLoad(self,filenames):
        self.linescan = nanonis.linescan()
        self.linescan.load(filenames)
        self.axMap.set_title(self.linescan.name[0]+' - '+self.linescan.name[-1], fontweight='bold')
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
            else:
                energyCut = event.xdata
                self.cutPlot(energyCut)


    def cutPlot(self, energy):
        bias = np.linspace(self.linescan.bias[0],self.linescan.bias[-1], len(self.linescan.bias))
        id = (abs(bias-energy)).argmin()
        print(id)
        self.axMap.plot([self.linescan.bias[id],self.linescan.bias[id]],[self.linescan.distance[0],self.linescan.distance[-1]])
        self.axCut.plot(self.linescan.distance, self.linescan.conductance[:,id], label=str(round(self.linescan.bias[id],2)))
        self.axCut.legend()
        self.saveCSV(self.linescan.distance, self.linescan.conductance[:,id])
        self.figure.canvas.draw_idle()
    
    def saveCSV(self, array1,array2):
        matrix = np.vstack((array1,array2))
        matrix = np.transpose(matrix)
        filename_mod = '/Volumes/Internal HD 400/PHD/OneDrive - Asociacion Cic Nanogune/STM_Stefano/Bi2Pd/2020-05-20-Mn/2020-05-Analysis/2020.05Analysis_general/cut.txt'
        with open(filename_mod, 'w') as csvfile:
            writer = csv.writer(csvfile)
            [writer.writerow(r) for r in matrix]
    
    #def locatePoints(self, filenames, filename):
    def update(self, val):
        self.im1.set_clim([self.smin.val,self.smax.val])
        self.figure.canvas.draw()
    
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





    







'''
plt.pcolormesh(lin.bias, lin.distance, conductance, vmin=0, vmax=2e-11)
plt.title(lin.name[0]+' - '+lin.name[-1], fontweight='bold')
plt.colorbar()
plt.xlabel('Energy [meV]')
plt.ylabel('Distance [nm]')
cbar = plt.colorbar(orientation='horizontal', pad=0.02, fraction=0.0468)
cbar.outline.set_visible(False)
cbar.locator = ticker.MaxNLocator(nbins=2)
cbar.update_ticks()
cbar.set_label('dI/dV [arb. units]', labelpad=-48)'''
