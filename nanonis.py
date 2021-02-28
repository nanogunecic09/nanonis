#nanonis file format analysis module
#nanoImaging group @ nanoGune

#from modules import functions
from numpy import arange, sqrt, array, linspace, zeros, rot90, flipud, fromfile, meshgrid, arange, fliplr, gradient, mean
from pandas import DataFrame, read_csv
from dateutil.parser import parse
from scipy import interpolate
import struct

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
        print(channels)
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
        elif 'SRX [AVG] (V)' in self.data:
            self.conductance = self.data['SRX [AVG] (V)']
            self.conductanceColumn = 'SRX [AVG] (V)'
        elif 'LIX 1 omega (A)' in self.data:
            self.conductance = self.data['LIX 1 omega (A)']
            self.conductanceColumn = 'LIX 1 omega (A)'
        if 'SRX2nd [AVG] (V)' in self.data:
            self.harmonic = self.data['SRX2nd [AVG] (V)']
        elif 'LIX 1 omega [AVG] (A)' in self.data:
            self.conductance = self.data['LIX 1 omega [AVG] (A)']
            self.conductanceColumn = 'LIX 1 omega [AVG] (A)'
        if 'Current (A)' in self.data:
            self.current = self.data['Current (A)']
            self.currentColumn = 'Current (A)'
        elif 'Current [AVG] (A)' in self.data:
            self.current = self.data['Current [AVG] (A)']
            self.currentColumn = 'Current [AVG] (A)'
        if 'X (m)' in self.header:
            self.x = float(self.header['X (m)'])
        if 'Y (m)' in self.header:
            self.y = float(self.header['Y (m)'])
        if 'Z (m)' in self.header:
            self.z = float(self.header['Z (m)'])
#        if self.header['Date']:
#            self.date = parse(self.header['Date'])

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

    def conductanceOffset(self, offset):
        self.data[self.conductanceColumn] = self.data[self.conductanceColumn]-offset
    
    def energyFind(self, energy):
        index = (abs(self.bias - energy)).idxmin()
        return index

    def normalizeTo(self, energy):
        index = self.energyFind(energy)
        self.conductance = self.conductance/self.conductance[index]

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

class linescan():

    def __init__(self):
        self.type = 'Linescan'

    def load(self, files):
        spectra = biasSpectroscopy()
        dummyCo = []
        dummyCu = []
        dummyNa = []
        dummyR = []
        spectra.load(files[0])
        x0,y0 = spectra.x,spectra.y
        spectra.load(files[-1])
        x1,y1 = spectra.x,spectra.y
        self.length = sqrt((x0-x1)**2+(y0-y1)**2)*1e9
        self.distance = linspace(self.length,0,len(files))
        self.bias = array(spectra.bias)*1e3
        for i in files:
            spectra.load(i)
            dummyCo.append(spectra.conductance)
            #dummyCu.append(spectra.current)
            dummyNa.append(spectra.name)
            #dummyR.append(spectra.bias[0]/spectra.current[0])
        self.conductance = array(dummyCo)
        self.current = array(dummyCu)
        self.name = array(dummyNa)
        self.resistance = array(dummyR)


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
        self.data, header, self.parameters, self.multiline, MLS = read3DS(filename)
        self.filename = filename
        '''if MLS:
            multi = list(self.multiline.keys())
            first = False
            self.sweep = []
            for i in range(len(multi[1])):
                self.sweep.extend(linspace(multi[0][i]), multi[1][i]), multi[5][i]))'''
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
            
    #def cutFind(self, value):
    #    index = (abs(self.sweep - value)).argmin()
    #    return index