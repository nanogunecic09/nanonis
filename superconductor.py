#Superconductor analysis module
#nanoImaging group @ nanoGune

import numpy as np
from . import distributions
from scipy import signal

# Evaluate the non-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array or value
#       * Delta is the superconducting gap at 0K

def BCS_curve(energy, Delta): 
        Delta = np.complex(Delta)
        density = np.sign(energy)*np.real(np.divide(energy,np.sqrt(np.power(energy,2)-np.power(Delta,2))))
        return density

# Evaluate the Dynes-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array
#       * Delta is the superconducting gap at 0K
#       * dynesParameter is the Dynes factor

def dynes_curve(energy, Delta, dynesParameter): 
        dynesParameter = np.complex(0,dynesParameter)
        density = np.sign(energy)*np.real(np.divide(energy-dynesParameter,np.sqrt(np.power(energy-dynesParameter,2)-np.power(Delta,2))))
        return density

# Evaluate the derivative Dynes-broadened DOS of a conventional s-wave superconductor
#       * energy is the energy array
#       * Delta is the superconducting gap at 0K
#       * dynesParameter is the Dynes factor

def dynes_curve_diff(energy, Delta, dynesParameter): 
        dynesParameter = np.complex(0,dynesParameter)
        density_de = -np.sign(energy)*np.real(np.divide(np.square(Delta),np.power(np.square(energy-dynesParameter)-np.square(Delta),1.5)))
        density_de = np.nan_to_num(density_de)
        return density_de