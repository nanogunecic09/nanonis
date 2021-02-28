import numpy as np
from math import pi

def gaussian (axis, std, mean=0):
 return np.divide(np.exp(-np.divide(np.square(axis-mean),(2*np.square(std)))),np.sqrt(2*pi*np.square(std)))

def cauchy (axis, scale, location=0):
 return np.multiply(np.divide(1,pi*scale),np.divide(np.square(scale),np.square(axis-location)+np.square(scale)))

def boltzmann (axis, temperature):
 return np.divide(1,np.exp(np.divide(axis,np.multiply(8.617E-5,temperature))))

def fermiDirac (axis, temperature):
 return np.divide(1,np.exp(np.divide(axis,np.multiply(8.617E-5,temperature)))+1)

def fermiDirac_diff(axis, temperature):
  return -np.divide(np.exp(np.divide(axis,np.multiply(8.617E-5,temperature))), np.multiply(np.multiply(8.617E-5,temperature),np.square(1+np.exp(np.divide(axis,np.multiply(8.617E-5,temperature))))))

def bose (axis, temperature):
 return np.divide(1,np.exp(np.divide(axis,np.multiply(8.617E-5,temperature)))-1)