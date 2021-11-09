#to get a list of measurement filenames given:
# path, staticname eg.: 'S211026_', indexes: (1,200) and estension (.dat default) 
def getfnames(path,staticname, idx,extension='.dat'):
    idxn = [f"{i:03}" for i in range(idx[0],idx[1]+1)]
    fnames = []
    for idx in idxn:
        fnames.append(path + staticname + idx + extension)
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