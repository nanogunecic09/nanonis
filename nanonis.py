#nanonis file format analysis module
#nanoImaging group @ nanoGune

#from modules import functions
from numpy import flip, arange, sqrt, array, linspace, zeros, rot90, flipud, fromfile, meshgrid, arange, fliplr, gradient, mean
import numpy as np
from pandas import DataFrame, read_csv
from dateutil.parser import parse
from scipy import interpolate
import struct
import colorcet as cc
import pandas as pd
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from distributions import fermiDirac, fermiDirac_diff
import deconvolution as deconv
import math
import glob
def readDATDec(filename, datatype):
    if datatype == 'dIdV':
        names = ['Bias', 'Conductance']
    if datatype == 'Cut':
        names = ['Distance(nm)', 'dI/dV (arb. units)']
    with open(filename) as f:
        data = read_csv(filename,sep=',', names=names)
        
    return data

def readGcutLS(didvfile,distancefile):
    conductance = []
    bias = []
    distance = []
    with open(didvfile) as f:
        dfdata = read_csv(didvfile, sep='\t', header=None)

    bias = dfdata.loc[:,0]
    conductance = dfdata.loc[:,1:dfdata.shape[1]:2].values
    conductance = rot90(conductance)
    conductance = fliplr(conductance)
    with open(distancefile) as f:
        dfdata = read_csv(distancefile, sep='\t', header=None)
    distance = dfdata.loc[:,0]
    return conductance, bias, distance

def readWSXMcloud(didvfile):
    conductance = []
    bias = []
    with open(didvfile) as f:
        dfdata = read_csv(didvfile, sep='\t', header=None)

    bias = dfdata.loc[:,0]
    conductance = dfdata.loc[:,1:dfdata.shape[1]:2].values
    conductance = rot90(conductance)
    conductance = fliplr(conductance)
    return bias,conductance

def readDAT(filename):
    header = dict()
    with open(filename) as f:
        for i, line in enumerate(f):
            if '[DATA]' in line:
                skipR = i+1
                break
            else: skipR = 0
            values = line.split('\t')
            if len(values) == 3:
                header[values[0]] = values[1]
    data = read_csv(filename, sep="\t", skiprows = skipR)
    return data, header

def read3DS(filename):
    header = dict()
    data = dict()
    param = dict()
    mls = dict()
    MLS = False
    with open(filename, 'rb') as f:
        for line in f:
            text = line.strip().decode()
            if 'Grid dim' in text:
                grid_raw = text.split('=')
                grid_raw = grid_raw[1].replace('"', '')
                grid_raw = grid_raw.split('x')
                header['XPixels'] = int(grid_raw[0])
                header['YPixels'] = int(grid_raw[1])
            if 'Grid settings' in text:
                gset_raw = text.split('=')
                gset_raw = gset_raw[1].split(';')
                header['X Offset (nm)'] = float(gset_raw[0])*1e9
                header['Y Offset (nm)'] = float(gset_raw[1])*1e9
                header['X Length (nm)'] = float(gset_raw[2])*1e9
                header['Y Length (nm)'] = float(gset_raw[3])*1e9
                header['Scan angle (deg)'] = float(gset_raw[4])
            if 'Filetype' in text:
                filetype = text.split('=')
                if filetype[1] == 'MLS':
                    MLS = True
            if 'Sweep Signal' in text:
                sweep_raw = text.split('=')
                header['Sweep Variable'] = sweep_raw[1].replace('"', '')
            if 'Fixed parameters' in text:
                nameparam = [] 
                parms_raw = text.split('=')
                parms_raw = parms_raw[1].replace('"', '')
                nameparam = parms_raw.split(';')
            if '# Parameters (4 byte)' in text:
                parms = text.split('=')
                parms_num = int(parms[1])
            if MLS:
                if 'Segment Start (V)' in text:
                    multiparams = []
                    mls_raw = text.split('=')
                    mls_params = mls_raw[0].split(',')
                    mls_values = mls_raw[1].split(';')
                    for i in range(len(mls_values)):
                        multiparams.append(mls_values[i].split(','))
                    for i in range(len(mls_params)):
                        params = []
                        for j in range(len(multiparams)):
                            params.append(float(multiparams[j][i]))
                        mls[mls_params[i]] = params
            if 'Experiment parameters' in text:
                parms_raw = text.split('=')
                parms_raw = parms_raw[1].replace('"', '')
                nameparam.extend(parms_raw.split(';'))
            if 'Points' in text:
                points = text.split('=')
                header['# Points'] = int(points[1])
            if 'Channels' in text:
                channels = []
                chann_raw = text.split('=')
                chann_raw = chann_raw[1].replace('"', '')
                channels = chann_raw.split(';')
            if ':HEADER_END:' in text:
                header_size = f.tell()
                break
    f = open(filename, 'rb')
    f.seek(header_size)
    griddata = fromfile(f, dtype='>f4')
    f.close()
    data_unso = griddata.reshape((header['YPixels'], header['XPixels'], parms_num+(len(channels)*header['# Points'])))
    for i in range(len(nameparam)):
        param[nameparam[i]] = data_unso[:,:,i]
    for i in range(len(channels)):
        data[channels[i]] = data_unso[:,:,parms_num+(i*header['# Points']):parms_num+(i*header['# Points'])+header['# Points']]
    return data, header, param, mls, MLS

def readSXM(filename):
    header = dict()
    data = dict()
    mparameters = dict()
    multipass = False
    with open(filename, 'rb') as f:
        for line in f:
            text = line.strip().decode()
            if ':SCAN_PIXELS:' in text:
                spixels = f.readline().strip().decode()
                spixels = spixels.split(' ')
                header['XPixels'] = int(spixels[0])
                header['YPixels'] = int(spixels[-1])
            if ':SCAN_RANGE:' in text:
                ssize = f.readline().strip().decode()
                ssize = ssize.split(' ')
                header['X Length (nm)'] = float(ssize[0])*1e9
                header['Y Length (nm)'] = float(ssize[-1])*1e9
            if ':SCAN_OFFSET:' in text:
                soffset = f.readline().strip().decode()
                soffset = soffset.split(' ')
                header['X Offset (nm)'] = float(soffset[0])*1e9
                header['Y Offset (nm)'] = float(soffset[-1])*1e9
            if ':SCAN_ANGLE:' in text:
                header['Scan Angle (deg)'] = float(f.readline().strip().decode())
            if ':SCAN_DIR:' in text:
                header['Scan direction'] = f.readline().strip().decode()
            if ':BIAS:' in text:
                header['Bias (V)'] = float(f.readline().strip().decode())
            if ':Z-CONTROLLER:' in text:
                f.readline()
                sfeedback = f.readline().strip().decode()
                sfeedback = sfeedback.split('\t')
                header['Feedback type'] = sfeedback[0]
                header['Feedback state'] = bool(sfeedback[1])
                ssetpoint = sfeedback[2].split(' ')
                header['Setpoint [{a}]'.format(a=ssetpoint[1])] = float(ssetpoint[0])
            if 'Multipass-Config' in text:
                multipass = True
                f.readline()
                mpass_tabs = f.readline().strip().decode().split('\t')
                mpenergies = []
                mpoffset = []
                mpsetpoint = []
                while len(mpass_tabs) == 9:
                    mpoffset.append(float(mpass_tabs[2]))
                    if mpass_tabs[4] == 'FALSE':
                        mpenergies.append(header['Bias (V)'])
                    else:
                        mpenergies.append(float(mpass_tabs[5]))
                    if mpass_tabs[6] == 'FALSE':
                        mpsetpoint.append(0)
                    else:
                        mpsetpoint.append(float(mpass_tabs[7]))
                    mpass_tabs = f.readline().strip().decode().split('\t')
            if 'DATA_INFO' in text:
                f.readline()
                data_tabs = f.readline().strip().decode().split('\t')
                channels = []
                while len(data_tabs) == 6:
                    if data_tabs[3] == 'both':
                        channels.append('{a} [{b}] (fwd)'.format(a=data_tabs[1],b=data_tabs[2]))
                        channels.append('{a} [{b}] (bwd)'.format(a=data_tabs[1],b=data_tabs[2]))
                    else:
                        channels.append('{a} [{b}]'.format(a=data_tabs[1],b=data_tabs[2]))
                    data_tabs = f.readline().strip().decode().split('\t')
            if ':SCANIT_END:' in text:
                break
    if multipass:
        passnum = []
        passoff = []
        passene = []
        passset = []
        for i in range(len(channels)):
            cha = channels[i]
            num = int(cha[cha.index('[P')+2:cha.index(']_')])
            if 'fwd' in cha:
                passnum.append((2*num)-1)
            else:
                passnum.append(2*num)
        for i in passnum:
            passoff.append(mpoffset[i-1])
            passene.append(mpenergies[i-1])
            passset.append(mpsetpoint[i-1])
        mparameters = {'Offset (m)': passoff, 'Energy (eV)':passene, 'Setpoint':passset}
    f = open(filename, 'rb')
    full = f.read()
    skipB = full.find(b'\x1A\x04')
    f.seek(skipB+2)
    for i in range(len(channels)):
        bindata = f.read(4*header['XPixels']*header['YPixels'])
        raw_data = zeros(header['XPixels']*header['YPixels'])
        for j in range(header['XPixels']*header['YPixels']):
            raw_data[j] = struct.unpack('>f', bindata[j*4: j*4+4])[0]
        raw_data = raw_data.reshape(header['YPixels'], header['XPixels'])
        raw_data = rot90(raw_data)
        if 'bwd' in channels[i]:
            raw_data = raw_data[::-1]
        raw_data = rot90(raw_data,3)
        if header['Scan direction'] == 'up':
            raw_data = flipud(raw_data)
        data[channels[i]] = raw_data
    f.close()
    return data, header, mparameters

class simpleScan():
#Falta implementar la lectura del setpoint

    def __init__(self):

        self.data = {'Z (m)':0}
        
    
    def load(self, fname):
        self.data, header, mparameters = readSXM (fname)
        self.filename = fname
        self.multipass = False
        dummy = self.filename.split("/")
        self.name = dummy[-1] 
        if 'Bias (V)' in header:
            self.bias = header['Bias (V)']
        if 'XPixels' in header:
            self.xpixels = header['XPixels']
        if 'YPixels' in header:
            self.ypixels = header['YPixels']
        if 'X Length (nm)' in header:
            self.xrange = header['X Length (nm)']
        if 'Y Length (nm)' in header:
            self.yrange = header['Y Length (nm)']
        if 'X Offset (nm)' in header:
            self.xoffset = header['X Offset (nm)']
        if 'Y Offset (nm)' in header:
            self.yoffset = header['Y Offset (nm)']
        if 'Scan Angle (deg)' in header:
            self.scanangle = header['Scan Angle (deg)']
        if 'Feedback type' in header:
            self.feedback = header['Feedback type']
        if 'Feedback state' in header:
            self.feedbackstate = header['Feedback state']
        if mparameters:
            self.mparameters = True
        else:
            self.mparameters = False
        if 'Offset (m)' in mparameters:
            self.mpoffset = mparameters['Offset (m)']
        if 'Energy (eV)' in mparameters:
            self.mpenergies = mparameters['Energy (eV)']
        if 'Setpoint' in mparameters:
            self.mpsetpoint = mparameters['Setpoint']
        if ':Multipass-Config:' in header:
            self.multipass = True
        
    
    #Define the real position in scan space.
    #The corners of the square are defined as
    #b--a
    #c--d
    #def absoulte(self):
        #self.X0, self.Y0 = meshgrid(linspace(-self.xrange/2,self.xrange/2,self.xpixels),linspace(-self.yrange/2,self.yrange/2,self.ypixels))
        
        #a1 = functions.rotatePoint(a0,self.scanangle)

class biasSpectroscopy():

    def __init__(self):

        self.data = {'Bias calc (V)':0}
    
    def load(self,fname):
        self.data, self.header = readDAT(fname)
        self.filename = fname
        dummy = self.filename.split("/")
        self.name = dummy[-1]
        if 'Bias calc (V)' in self.data:
            self.bias = self.data['Bias calc (V)']
        elif 'Bias (V)' in self.data:
            self.bias = self.data['Bias (V)']
        #if 'X (m)' in self.data:
        #    self.xsweep = self.data['X (m)']
        #if 'Y (m)' in self.data:
        #    self.ysweep = self.data['Y (m)']
        if 'Z (m)' in self.data:
            self.zsweep = self.data['Z (m)']
        if 'Z [bwd] (m)' in self.data:
            self.zsweepb = self.data['Z [bwd] (m)']
        if 'SRX (V)' in self.data:
            self.conductance = self.data['SRX (V)']
            self.conductanceColumn = 'SRX (V)'
        if 'LI Demod 1 X (A)' in self.data:
            self.conductance = self.data['LI Demod 1 X (A)']
            self.conductanceColumn = 'LI Demod 1 X (A)'
        if 'LI Demod 1 X [AVG] (A)' in self.data:
            self.conductance = self.data['LI Demod 1 X [AVG] (A)']
            self.conductanceColumn = 'LI Demod 1 X [AVG] (A)'
        if 'SRY (V)' in self.data:
            self.sry = self.data['SRY (V)']
        elif 'SRX [AVG] (V)' in self.data:
            self.conductance = self.data['SRX [AVG] (V)']
            self.conductanceColumn = 'SRX [AVG] (V)'
        elif 'LIX 1 omega (A)' in self.data:
            self.conductance = self.data['LIX 1 omega (A)']
            self.conductanceColumn = 'LIX 1 omega (A)'
        if 'SRX [bwd] (V)' in self.data:
            self.conductanceb = self.data['SRX [bwd] (V)']
            self.conductancebColumn = 'SRX [bwd] (V)'
        if 'SRY [bwd] (V)' in self.data:
            self.sryb = self.data['SRY [bwd] (V)']
        if 'SRX2nd [AVG] (V)' in self.data:
            self.harmonic = self.data['SRX2nd [AVG] (V)']
        elif 'LIX 1 omega [AVG] (A)' in self.data:
            self.conductance = self.data['LIX 1 omega [AVG] (A)']
            self.conductanceColumn = 'LIX 1 omega [AVG] (A)'
        if 'LIX n omega [AVG] (A)' in self.data:
            self.conductance2 = self.data['LIX n omega [AVG] (A)']
            self.conductance2Column = 'LIX n omega [AVG] (A)'
        if 'Current (A)' in self.data:
            self.current = self.data['Current (A)']
            self.currentColumn = 'Current (A)'
        elif 'Current [AVG] (A)' in self.data:
            self.current = self.data['Current [AVG] (A)']
            self.currentColumn = 'Current [AVG] (A)'
        elif 'Current [AVG]' in self.data:
            self.current = self.data['Current [AVG] (A)']
            self.currentColumn = 'Current [AVG] (A)'
        if 'Current [bwd] (A)' in self.data:
            self.currentb = self.data['Current [bwd] (A)']
            self.currentbColumn = 'Current [bwd] (A)'
        if 'X (m)' in self.header:
            self.x = float(self.header['X (m)'])
        if 'Y (m)' in self.header:
            self.y = float(self.header['Y (m)'])
        if 'Z (m)' in self.header:
            self.z = float(self.header['Z (m)'])
        if 'Lock-In Signal (V)' in self.data:
            self.conductance = self.data['Lock-In Signal (V)']
        if 'Ext. VI 1>7270 Modulation (V)' in self.header:
            self.modamp = float(self.header['Ext. VI 1>7270 Modulation (V)'])
        if 'Ext. VI 1>7270 Sensitivity (V)' in self.header:
            self.sens = float(self.header['Ext. VI 1>7270 Sensitivity (V)' ])
        if 'Current>Gain' in self.header:
            self.gain = 10**float(self.header['Current>Gain'][-1])
        if 'Bias>Calibration (V/V)' in self.header:
            self.biascal = float(self.header['Bias>Calibration (V/V)'])
#        if self.header['Date']:
#            self.date = parse(self.header['Date'])
        if 'LI Demod 1 X (A)' in self.data:
            self.conductance = self.data['LI Demod 1 X (A)']
        if 'LI Demod 1 X [bwd] (A)' in self.data:
            self.conductance_bwd = self.data['LI Demod 1 X [bwd] (A)']
        if 'LI Demod 1 X [AVG] (A)' in self.data:
            self.conductance = self.data['LI Demod 1 X [AVG] (A)']
        if 'LI Demod 1 X [AVG] [bwd] (A)' in self.data:
            self.conductance_bwd = self.data['LI Demod 1 X [AVG] [bwd] (A)']
        if 'LI Demod 1 R (V)' in self.data:
            self.conductance = self.data['LI Demod 1 R (V)']
        if 'LI Demod 1 R (A) [bwd]' in self.data:
            self.conductance_bwd = self.data['LI Demod 1 R (A) [bwd]']
        if 'Bias_VI (V)' in self.data:
            self.biasVI_b = self.data['Bias_VI (V)']
        if 'Bias_VI [bwd] (V)' in self.data:
            self.biasVI_f = self.data['Bias_VI [bwd] (V)']
        if 'Input 2 (V)' in self.data:
            self.biasVI_b = self.data['Input 2 (V)']
        if 'Input 2 [bwd] (V)' in self.data:
            self.biasVI_f = self.data['Input 2 [bwd] (V)']
        if 'Input 2 [AVG] [bwd] (V)' in self.data:
            self.biasVI_b = self.data['Input 2 [AVG] [bwd] (V)']
        if 'Input 2 [AVG] (V)' in self.data:
            self.biasVI_f = self.data['Input 2 [AVG] (V)']
    def biasOffset(self, offset):
        self.data['Bias calc (V)'] = self.data['Bias calc (V)']-offset

    def normalizeRange(self, range): #normalize data given an energy range
        index = []
        index.append(self.energyFind(range[1]))
        index.append(self.energyFind(range[0]))
        conductanceCut = self.conductance[index[0]:index[1]]
        avg = mean(conductanceCut)
        self.conductance[:] = self.conductance[:]/avg
    #def currentOffset(self, offset):
    #    self.data[self.currentColumn] = self.data[self.currentColumn]-offset


    def normalizeRange_symm(self, range): #normalize data given an energy range using both positive and negative baselines
        index = []
        index.append(self.energyFind(range[1]))
        index.append(self.energyFind(range[0]))
        conductanceCut_neg = self.conductance[index[0]:index[1]]
        index = []
        index.append(self.energyFind(-range[1]))
        index.append(self.energyFind(-range[0]))
        conductanceCut_pos = self.conductance[index[1]:index[0]]        
        avg = mean(np.concatenate((conductanceCut_neg,conductanceCut_pos)))
        self.conductance[:] = self.conductance[:]/avg
        return conductanceCut_neg,conductanceCut_pos
    #def currentOffset(self, offset):
    #    self.data[self.currentColumn] = self.data[self.currentColumn]-offset

    def conductanceOffset(self, offset):
        self.data[self.conductanceColumn] = self.data[self.conductanceColumn]-offset
    
    def energyFind(self, energy):
        index = (abs(self.bias - energy)).idxmin()
        return index

    def normalizeTo(self, energy):
        index = self.energyFind(energy)
        self.conductance = self.conductance/self.conductance[index]

    def calcdidv(self,biascal):
        didv=2*self.conductance/2.5*self.sens/(self.modamp*biascal)/self.gain #the origin of the factor 2 is unknown - perhaps rms related?
        return didv

    def biasCalibration(self):
        self.bias = self.bias*1.0312

    #def currDiff(self):
    #    currDiff = -gradient(self.current)
    #    return corrConductance

    #def currNormalize(self, energy):
    #    index = self.energyFind(energy)
    #    corrConductance = currDiff()
    #    currDiffnorm = currDiff/currDiff[index]
    #    return currDiffnorm

    def linearize(self, factor):
        difference = abs(self.bias[1]-self.bias[0])
        dummy = []
        for i in range(len(self.bias)-1):
            dummy = abs(self.bias[i+1]-self.bias[i])
            if dummy < difference:
                difference = dummy
        self.biasLin = arange(min(self.bias),max(self.bias),difference/factor)
        interp = interpolate.interp1d(self.bias, self.conductance)
        self.conductanceLin = interp(self.biasLin)

    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return idx
    ##### Deconvolution #####
    def dynesDeconvolute(self,gap=1.30e-3, temperature=1.248, dynesParameter=40e-6, energyR=6E-3, spacing=50e-6,x_min=-3.0E-3,x_max=3.0E-3,N=1000, window=3,order=1,n=1000):
        self.bias_dec, self.conductance_dec = deconv.dynesDeconvolute(self.bias,self.conductance,gap, temperature, dynesParameter, energyR, spacing ,x_min,x_max,N, window,order,n)
        self.bias_dec = pd.Series(np.flip(self.bias_dec))
        self.conductance_dec = pd.Series(self.conductance_dec)

    def dynesDeconvolute_nof(self,gap=1.25e-3, temperature=1.7, dynesParameter=40E-6, energyR=6E-3, spacing = 56e-6,x_min=-2.0E-3,x_max=2.0E-3,N=1000,):
        self.bias_dec, self.conductance_dec = deconv.dynesDeconvolute_nof(self.bias,self.conductance, gap, temperature, dynesParameter, energyR, spacing,x_min,x_max,N)
        self.bias_dec = pd.Series(np.flip(self.bias_dec))
        self.conductance_dec = pd.Series(self.conductance_dec)

    def dec_normalizeTo(self,energy):
        index = (abs(self.bias_dec - energy)).idxmin()
        self.conductance_dec = self.conductance_dec/self.conductance_dec[index]

    def R_calc(self):
        off_f = self.biasVI_f[np.abs(self.bias-0).argmin()]
        off_b = self.biasVI_b[np.abs(self.bias-0).argmin()]
        self.biasVI_b = self.biasVI_b-off_b
        self.biasVI_f = self.biasVI_f-off_f
        self.current_lin = np.linspace(self.current.min(),self.current.max(),self.bias.shape[0])
        self.resistance_b = self.biasVI_b*1e6/(self.bias*(self.current.max()*1e9/(self.bias.max()*1e3)))
        self.resistance_f = self.biasVI_f*1e6/(self.bias*(self.current.max()*1e9/(self.bias.max()*1e3)))

    def current_cal(self):
        self.curr = np.flip(1e9*np.linspace(self.current.min(),self.current.max(),self.bias.shape[0]))
class linescan():

    def __init__(self):
        self.type = 'Linescan'

    def load(self, files,normalize=False,normalize_range = [3e-3,4e-3]):
        spectra = biasSpectroscopy()
        dummyCo = []
        dummyCu = []
        dummyNa = []
        dummyR = []
        dummyZ = []
        dummyTime= []
        spectra.load(files[0])
        self.x0,self.y0 = spectra.x,spectra.y
        spectra.load(files[-1])
        self.x1,self.y1 = spectra.x,spectra.y
        self.length = sqrt((self.x0-self.x1)**2+(self.y0-self.y1)**2)*1e9
        self.distance = linspace(self.length,0,len(files))
        self.bias = np.flip(array(spectra.bias))
        for i in files:
            spectra.load(i)
            if normalize == True:
                spectra.normalizeRange_symm(normalize_range)
            dummyCo.append(spectra.conductance)
            dummyCu.append(spectra.current)
            dummyNa.append(spectra.name)
            dummyR.append(spectra.bias[0]/spectra.current[0])
            dummyZ.append(spectra.z)
            # dummyTime.append(spectra.header['Date'])
        self.conductance = array(dummyCo)
        self.conductance = np.fliplr(self.conductance)
        self.current = array(dummyCu)
        self.name = array(dummyNa)
        self.resistance = array(dummyR)
        self.z = array(dummyZ)
        self.Time = dummyTime

    def distanceOffset(self, offset):
        self.distance = self.distance-offset
    
    def biasOffset(self, offset):
        self.bias = self.bias-offset

    def biasOffset_full(self,offset):
        self.bias = self.bias-offset
        

    def positionFind(self, position):
        index = (abs(self.distance - position)).argmin()
        return index
    
    def energyFind(self, energy):
        index = (abs(self.bias - energy)).argmin()
        return index

    def positionCut(self, position):
        index = (abs(self.distance - position)).argmin()
        return self.conductance[index,:]
        
    def energyCut(self, energy):
        index = (abs(self.bias - energy)).argmin()
        return self.conductance[:,index]

    def normalizeTo(self, energy):
        index = self.energyFind(energy)
        print(index)
        for i in range(len(self.name)):
            self.conductance[i][:] = self.conductance[i][:]/self.conductance[i][index]

    def normalizeRange(self, E_range): #normalize data given an energy range
        index = []
        index.append(self.energyFind(E_range[0]))
        index.append(self.energyFind(E_range[1]))
        for i in range(len(self.name)):
            conductanceCut = self.conductance[i][index[0]:index[1]]
            avg = mean(conductanceCut)
            self.conductance[i][:] = self.conductance[i][:]/avg

    def normalizeRange_symm(self, E_range): #normalize data given an energy range
        index = []
        index.append(self.energyFind(E_range[0]))
        index.append(self.energyFind(E_range[1]))
        for i in range(len(self.name)):
            conductanceCut = self.conductance[i][index[0]:index[1]]
            avg = mean(conductanceCut)
            self.conductance[i][:] = self.conductance[i][:]/avg


    def hand_normalization(self,path_values):
        values = pd.read_csv(path_values,header=None)
        values = values.to_numpy()
        for i in range(len(self.name)):
            self.conductance[i][:] = self.conductance[i][:]/values[i]
    
    def normalizeRange_posneg(self,negE_range,posE_range):

        index = []
        index.append(self.energyFind(-negE_range[0]))
        index.append(self.energyFind(-negE_range[1]))
        zero_idx = self.energyFind(0)
        for i in range(len(self.name)):
            conductanceCut = self.conductance[i][index[0]:index[1]]
            avg = mean(conductanceCut)
            self.conductance[i][0:zero_idx] = self.conductance[i][0:zero_idx]/avg

        index = []
        index.append(self.energyFind(-posE_range[0]))
        index.append(self.energyFind(-posE_range[1]))
        for i in range(len(self.name)):
            conductanceCut = self.conductance[i][index[0]:index[1]]
            avg = mean(conductanceCut)
            self.conductance[i][zero_idx:] = self.conductance[i][zero_idx:]/avg

    def biasCalibration(self):
        self.bias = self.bias*1.0312

    ### Perform deconvolution extending the spectra with N point
    def deconvolution_nof(self,gap=1.37e-3, temperature=1.3, dynesParameter=40e-6, energyR=8e-3, spacing=35e-6,x_min=-4E-3,x_max=4E-3,N=300,normalizeE = 3e-3):
        self.conductance_dec = np.zeros((self.distance.shape[0],int(math.ceil(energyR*2/spacing))))
        for i in range(self.conductance.shape[0]):
            self.bias_dec, self.conductance_dec[i,:] = deconv.dynesDeconvolute_nof(self.bias,self.conductance[i,:],gap, temperature, dynesParameter, energyR, spacing,x_min,x_max,N)
        # normalize
        self.bias_dec = self.bias_dec
        for i in range(0,self.conductance_dec.shape[0]):
            self.conductance_dec[i,:] = self.conductance_dec[i,:]/self.conductance_dec[i,abs(self.bias_dec-normalizeE).argmin()]
        #flip it
        self.conductance_dec = np.fliplr(self.conductance_dec)

    ### Perform deconvolution extending the spectra with N point and applying a Savitzky–Golay filter to the data
    def deconvolution(self,gap=1.37e-3, temperature=1.3, dynesParameter=40e-6, energyR=8e-3, spacing=35e-6,x_min=-4E-3,x_max=4E-3,N=300, window=15,order=2,n=2000,normalizeE = 3e-3):
        self.conductance_dec = np.zeros((len(self.distance),int(math.ceil(energyR*2/spacing))))
        for i in range(self.conductance.shape[0]):
            self.bias_dec, self.conductance_dec[i,:] = deconv.dynesDeconvolute(self.bias,self.conductance[i,:],gap, temperature, dynesParameter, energyR, spacing,x_min,x_max,N, window,order,n)
        self.bias_dec = self.bias_dec
        # normalize
        # for i in range(0,self.conductance_dec.shape[0]):
        #     self.conductance_dec[i,:] = self.conductance_dec[i,:]/self.conductance_dec[i,abs(self.bias_dec-normalizeE).argmin()]
        #flip it
        self.conductance_dec = np.fliplr(self.conductance_dec)

    def energyFind_dec(self, energy):
        index = (abs(self.bias_dec - energy)).argmin()
        return index

    def normalizeTo_dec(self, energy):
        index = self.energyFind_dec(energy)
        for i in range(len(self.name)):
            self.conductance_dec[i][:] = self.conductance_dec[i][:]/self.conductance_dec[i][index]

    def normalizeRange_dec(self, E_range): #normalize data given an energy range
        index = []
        index.append(self.energyFind_dec(E_range[0]))
        index.append(self.energyFind_dec(E_range[1]))
        for i in range(len(self.name)):
            conductanceCut = self.conductance_dec[i][index[0]:index[1]]
            avg = mean(conductanceCut)
            self.conductance_dec[i][:] = self.conductance_dec[i][:]/avg

class Zapproach():
        def __init__(self):
            self.type = 'Linescan'

        def load(self, files):
            self.spectra = biasSpectroscopy()
            dummyCo = []
            dummyCu = []
            dummyNa = []
            dummyR = []
            dummyI0 = []
            dummyOff = []
            dummyZ = []
            self.spectra.load(files[0])
            self.bias = self.spectra.bias
            self.z0 = float(self.spectra.header['Z (m)'])
            for i in files:
                self.spectra.load(i)
                dummyCo.append(self.spectra.conductance)
                dummyCu.append(self.spectra.current)
                dummyNa.append(self.spectra.name)
                dummyR.append(self.spectra.bias[0]/self.spectra.current[0])
                dummyOff.append(float(self.spectra.header['Bias Spectroscopy>Z offset (m)']))
                dummyZ.append(float(self.spectra.header['Z (m)'])-self.z0)
                dummyI0.append(self.spectra.current[0])
            self.I0 = array(dummyI0)
            self.conductance = fliplr(array(dummyCo))
            self.current = array(dummyCu)
            self.name = array(dummyNa)
            self.resistance = array(dummyR)
            self.Zoff = array(dummyOff)
            self.Z = array(dummyZ)


        def distanceOffset(self, offset):
            self.distance = self.distance-offset

        def biasOffset(self, offset):
            self.bias = self.bias-offset

        def positionFind(self, position):
            index = (abs(self.distance - position)).argmin()
            return index

        def energyFind(self, energy):
            index = (abs(self.bias - energy)).argmin()
            return index

        def positionCut(self, position):
            index = (abs(self.distance - position)).argmin()
            return self.conductance[index,:]

        def energyCut(self, energy):
            index = (abs(self.bias - energy)).argmin()
            return self.conductance[:,index]

        def normalizeTo(self, energy):
            index = self.energyFind(energy)
            for i in range(len(self.name)):
                self.conductance[i][:] = self.conductance[i][:]/self.conductance[i][index]

        def normalizeRange(self, E_range): #normalize data given an energy range
            index = []
            index.append(self.energyFind(E_range[1]))
            index.append(self.energyFind(E_range[0]))
            for i in range(len(self.name)):
                conductanceCut = self.conductance[i][index[0]:index[1]]
                avg = mean(conductanceCut)
                self.conductance[i][:] = self.conductance[i][:]/avg

class grid():
#Falta hacer bien el MLS
    def __init__(self):
        self.type = 'Grid'

    def load(self, filename):
        self.data, self.header, self.parameters, self.multiline, MLS = read3DS(filename)
        self.filename = filename
        self.bias = linspace(self.parameters['Sweep Start'][0][0],self.parameters['Sweep End'][0][0],self.header['# Points'])
        '''if MLS:
            multi = list(self.multiline.keys())
            first = False
            self.sweep = []
            for i in range(len(multi[1])):
                self.sweep.extend(linspace(multi[0][i]), multi[1][i]), multi[5][i]))'''
        if 'X Length (nm)' in self.header:
            self.xrange = self.header['X Length (nm)']
        if 'Y Length (nm)' in self.header:
            self.yrange = self.header['Y Length (nm)']
        if 'X Offset (nm)' in self.header:
            self.xoffset = self.header['X Offset (nm)']
        if 'Y Offset (nm)' in self.header:
            self.yoffset = self.header['Y Offset (nm)']
        if 'Scan Angle (deg)' in self.header:
            self.scanangle = self.header['Scan Angle (deg)']


    #def cutFind(self, value):
    #    index = (abs(self.sweep - value)).argmin()
    #    return index

class linescan3ds():

    def __init__(self):
        self.type = 'Linescan'

    def load(self, filename):
        self.data, self.header, self.parameters, self.multiline, MLS = read3DS(filename)
        self.length = self.header['X Length (nm)']
        self.distance = linspace(0,self.length,num=self.header['# Points'])
        self.bias = flip(linspace(self.parameters['Sweep Start'][0][0]*1e3,self.parameters['Sweep End'][0][0]*1e3,num=self.header['# Points']))
        self.conductance = self.data['LIX 1 omega (A)'][0,:,:]
        self.name = filename.split("/")


    def distanceOffset(self, offset):
        self.distance = self.distance-offset
    
    def biasOffset(self, offset):
        self.bias = self.bias-offset

    def positionFind(self, position):
        index = (abs(self.distance - position)).argmin()
        return index
    
    def energyFind(self, energy):
        index = (abs(self.bias - energy)).argmin()
        return index

    def positionCut(self, position):
        index = (abs(self.distance - position)).argmin()
        return self.conductance[index,:]
        
    def energyCut(self, energy):
        index = (abs(self.bias - energy)).argmin()
        return self.conductance[:,index]

    def normalizeTo(self, energy):
        index = self.energyFind(energy)
        for i in range(len(self.name)):
            self.conductance[i][:] = self.conductance[i][:]/self.conductance[i][index]

    def normalizeRange(self, E_range): #normalize data given an energy range
        index = []
        index.append(self.energyFind(E_range[1]))
        index.append(self.energyFind(E_range[0]))
        for i in range(len(self.name)):
            conductanceCut = self.conductance[i][index[0]:index[1]]
            avg = mean(conductanceCut)
            self.conductance[i][:] = self.conductance[i][:]/avg

