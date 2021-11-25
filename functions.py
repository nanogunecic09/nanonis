import numpy as np
import distributions
import nanonis
import scipy.signal as signal
from lmfit.models import VoigtModel
from scipy import constants as const
import matplotlib.pyplot as plt

def filterG(data, points=40, p=0.5, sigma=10):
	window = distributions.gaussian(np.linspace(-50,50,points), sigma)
	window = window/max(window)
	filtered = signal.fftconvolve(window, data)
	filtered = (np.average(data) / np.average(filtered)) * filtered
	filtered = np.roll(filtered,-int(points/2))
	return filtered[:-points]

def filterC(data, points=40, p=0.5, sigma=10):
	window = distributions.cauchy(np.linspace(-50,50,points), sigma)
	window = window/max(window)
	filtered = signal.fftconvolve(window, data)
	filtered = (np.average(data) / np.average(filtered)) * filtered
	filtered = np.roll(filtered,-int(points/2))
	return filtered[:-points]

def normalize(data, range=1.7E-3):
	indices = np.where(np.logical_or(data.bias>=range, data.bias<=-range))
	normal_state = np.mean(data.conductance[indices[0]])
	return data.conductance/normal_state

def getOffset(data):
	indices = np.where(np.logical_and(data.bias>=-7.5E-4, data.bias<=7.5E-4))
	start_ind = indices[0][0]
	end_ind = indices[0][-1]
	p = np.polyfit(data.bias[start_ind:end_ind],data.current[start_ind:end_ind],7)
	p2 = np.polyder(p,2)
	p1d1 = np.poly1d(p)
	p1d2 = np.poly1d(p2)
	superpoli = p1d1-p1d2
	return min(abs(superpoli.r))

def getOffset2(data):
	indices = np.where(np.logical_and(data.bias>=-7.5E-4, data.bias<=7.5E-4))
	start_ind = indices[0][0]
	end_ind = indices[0][-1]
	bias = np.array(data.bias[start_ind:end_ind])
	current = np.array(data.current[start_ind:end_ind])
	zero = np.argmin(abs(current))
	bias1 = np.array(bias[:zero])
	bias2 = np.array(bias[zero:])
	curr1 = np.array(current[:zero])
	curr2 = np.array(current[zero:])
	with plt.style.context('ggplot'):
		ax1=plt.subplot(211)
		ax1.plot(bias1, curr1,'*')
		ax1.plot(abs(bias2)+0.6E-4, abs(curr2),'*')
		ax2 = plt.subplot(212)
		ax2.plot(np.correlate(curr1,abs(curr2),'same'),'*')
	plt.show()


def fitVoigt(data, filtered, indices):
	gmodel = VoigtModel()
	center = []
	height = []
	for i in indices:
		index = (np.abs(data.bias-filtered.bias[i])).argmin()
		if data.bias[index]>=-1.4E-3 and data.bias[index]<=1.4E-3:
			indexO = (np.abs(data.bias-data.bias[index]-1.5E-4)).argmin()
			indexF = (np.abs(data.bias-data.bias[index]+1.5E-4)).argmin()
			xdata = data.bias[indexO:indexF]
			ydata = data.conductance[indexO:indexF]
			params = gmodel.make_params(amplitude=ydata.max(),
                            center=xdata.mean(),
                            sigma=xdata.std())
			result = gmodel.fit(ydata, params, x=xdata)
			parameters = result.params.valuesdict()
			center = np.append(center, parameters['center'])
			height = np.append(height,parameters['height'])
	if len(center) == 2:
		suma = {'PP1':center[0],'PP2':center[1],'HP1':height[0],'HP2':height[1],'SPEC':data.filename}
		return suma

def rotatePoint(point, angle):
	x = point[0]*np.cos(np.deg2rad(angle))-point[1]*np.sin(np.deg2rad(angle))
	y = point[0]*np.sin(np.deg2rad(angle))+point[1]*np.cos(np.deg2rad(angle))
	newPoint = [x,y]
	return  newPoint

def coherence(x,D,Delta,T):
	a = np.exp(-x/(np.sqrt(const.hbar*D/(Delta*const.e))))
	b = np.exp(-x/(np.sqrt(const.hbar*D/(2*np.pi*const.k*T))))
	return a*b