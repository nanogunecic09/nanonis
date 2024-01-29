import matplotlib.pyplot as plt
import pickle
import time
import numpy as np

import ast
from matplotlib.colors import LinearSegmentedColormap
import matplotlib

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
#to get a list of measurement filenames given:
# path, staticname eg.: 'S211026_', indexes: (1,200) and estension (.dat default) 
def getfnames(path,staticname, idx,extension='.dat'):
    fnames = []
    for i in idx:
        fnames.append(path + staticname + '{:03}'.format(i) + extension)
    return fnames

def getfnames_5(path,staticname, idx,extension='.dat'):
    fnames = []
    for i in idx:
        fnames.append(path + staticname + '{:05}'.format(i) + extension)
    return fnames

from scipy.signal import spectral
import deconvolution as deconv
import nanonis
spectra = nanonis.biasSpectroscopy()
#output the average of the conductance of list of filenames (spectra)
def avg_cond(filenames):
    conductance_avg = 0
    for f in filenames:
        spectra.load(f)
        spectra.normalizeRange([3e-3,4e-3])
        conductance_avg = conductance_avg + spectra.conductance
    conductance_avg = conductance_avg/len(filenames)
    return spectra.bias,conductance_avg

def avg_cond_decon(filenames):
    conductance_avg = 0
    for f in filenames:
        spectra.load(f)
        spectra.normalizeRange([3e-3,4e-3])
        bias, spectra.conductance = deconv.dynesDeconvolute_nof(spectra.bias,spectra.conductance,gap=1.4e-3,temperature=1.7, dynesParameter=40E-6, energyR=6E-3, spacing = 53.5e-6,x_min=-2.0E-3,x_max=2.0E-3,N=100)
        conductance_avg = conductance_avg + spectra.conductance
    conductance_avg = conductance_avg/len(filenames)
    return bias,conductance_avg/conductance_avg[0]


def set_size_cm(w,h, ax=None):
    """ w, h: width, height in cm """
    cm = 1/2.54
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w*cm)/(r-l)
    figh = float(h*cm)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 
def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)
    

def didv(axs):
    if type(axs) == type(np.zeros(2)):
        for ax in axs:
            ax.set_xlabel('Bias (mV)',fontname="Arial")
            ax.set_ylabel('dI/dV '+r'(G$_N$)',fontname="Arial")
            ax.tick_params(axis='both',direction='in')
    else:
        axs.set_xlabel('Bias (mV)',fontname="Arial")
        axs.set_ylabel('dI/dV '+r'(G$_N$)',fontname="Arial")
        axs.tick_params(axis='both',direction='in')

def didv_p():
        plt.xlabel('Bias (mV)',fontname="Arial")
        plt.ylabel('dI/dV '+r'(G$_N$)',fontname="Arial")
        plt.tick_params(axis='both',direction='in')

def didv_dec(axs):
    if type(axs) == type(np.zeros(2)):
        for ax in axs:
            ax.set_xlabel('Bias (mV)')
            ax.set_ylabel('dI/dV dec. '+r'(G$_N$)')
            ax.tick_params(axis='both',direction='in')
    else:
        axs.set_xlabel('E-E'+r'$_F$ (meV)')
        axs.set_ylabel('dI/dV dec. '+r'(G$_N$)')
        axs.tick_params(axis='both',direction='in')

def inner(axs):
    if type(axs) == type(np.zeros(2)):
        for axn in axs:
            axn.tick_params(axis='x', direction='in')
            axn.tick_params(axis='y', direction='in')
    else:
            axs.tick_params(axis='x', direction='in')
            axs.tick_params(axis='y', direction='in')

def timing(i,t0,cycles):
    if i == 1:
        #to predict job time
        t1 = time.time()
        cycle = (t1-t0)/60
        total = np.round(cycle*cycles,3)
        if total <1:
            print('1 cycle:',np.round(cycle,3),'minutes --- Total time:',np.round(total*60,2),'seconds')
        else:
            print('1 cycle:',np.round(cycle,3),'minutes --- Total time:',total,'minutes')

def energyFind(bias, energy):
    index = (abs(bias - energy)).argmin()
    return index



def data_smooth(x,y,order=1):
    interp_func = interp1d(x, y, kind='cubic')
    new_x = np.linspace(x.min(),x.max(),2000)
    int_y = interp_func(new_x)
    yy = gaussian_filter(int_y,order)
    return new_x,yy



def cmap_fromLut(fname): # converts a .lut file given its path to a LinearSegmentedColormap
    # locate control points number for each color [blue,green,red] and store data
    control_points = []
    data = []
    with open(fname, "r") as f:
        for line in f:
            data.append(line.strip())
            if 'Number of Control Points' in line:
                split = line.split(':')
                control_points.append(int(split[1].strip()))
    # locate the
    location = []
    n=0
    for i in data:
        if 'Info' in i:
            location.append(n)
        n+=1
    # separate and store the control points
    points_B = data[location[0]+2:location[0]+2+control_points[0]]
    points_G = data[location[1]+2:location[1]+2+control_points[1]]
    points_R = data[location[2]+2:location[2]+2+control_points[2]]

    def convert(list): # to convert from string to array format
        A = []
        for i in list:
            A.append(ast.literal_eval(i.split(':')[1].strip()))
        A = np.array(A)
        sorted_indices = np.argsort(A[:,0])
        arr = A[sorted_indices]
        return arr

    points_B = convert(points_B)
    points_G = convert(points_G)
    points_R = convert(points_R)
    #invert the color map to match the original
    points_B[:,1] = 255-points_B[:,1]
    points_G[:,1] = 255-points_G[:,1]
    points_R[:,1] = 255-points_R[:,1]

    # Create the colormap
    cmap_lut = LinearSegmentedColormap('lut_colormap', {
        'red':   [(x/255.0, y/255.0, y/255.0) for x, y in points_R],
        'green': [(x/255.0, y/255.0, y/255.0) for x, y in points_G],
        'blue':  [(x/255.0, y/255.0, y/255.0) for x, y in points_B]
    })
    return cmap_lut


def export_colormap(colormap, filename):
    cmaplist = [colormap(i) for i in range(colormap.N)]
    cmaplist[0] = (1.0,1.0,1.0,1.0)
    colormap = matplotlib.colors.LinearSegmentedColormap.from_list('mcm',cmaplist, colormap.N)

    with open(filename, 'w') as file:
        file.write("WSxM file copyright UAM\n")
        file.write("New Format Palette. 2001\n")
        file.write("Image header size: 1143\n\n")
        
        def write_color_points(color_name, color_points):
            file.write(f"[{color_name} Info]\n")
            for i, point in enumerate(color_points):
                file.write(f"    Control Point {i}: ({int(point[0]*255)} , {255-int(point[1]*255)})\n")
            file.write(f"    Number of Control Points: {len(color_points)}\n\n")
        
        write_color_points("Blue", colormap._segmentdata['blue'])
        write_color_points("Green", colormap._segmentdata['green'])
        write_color_points("Red", colormap._segmentdata['red'])
        
        file.write("[Palette Generation Settings]\n")
        file.write("    Derivate Mode for the last blue Point: Automatic\n")
        file.write("    Derivate Mode for the last green Point: Automatic\n")
        file.write("    Derivate Mode for the last red Point: Automatic\n")
        file.write("    Is there a particular palette index colored?: No\n")
        file.write("    Smooth Blue: No\n")
        file.write("    Smooth Green: No\n")
        file.write("    Smooth Red: No\n\n")
        
        file.write("[Header end]")

