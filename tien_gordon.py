import numpy as np
import matplotlib.pyplot as plt
import nanonis
import scipy.constants as const
import scipy as sc
import useful as uf
import scipy.signal as spsg
import scipy.interpolate as spi
class TG():
	"""
	Thien-Gordon (TG) simulator for photon-assisted tunneling in STM spectroscopy.

	This class computes a Thien–Gordon (Tien-Gordon) map, i.e., the modulation
	of tunneling conductance under an applied RF signal, based on the
	experimental dI/dV spectrum of a superconducting junction.
	"""
	
	def __init__(self, f=20, V_RF=1e-3,V_px=400,R_load=1e6):
		"""
		Initialize TG parameters.

		Parameters:
		-----------
		f : float
			RF frequency in GHz.
		V_RF : float
			Amplitude of the applied RF voltage (in Volts).
		V_max : float
			Maximum bias voltage for interpolation grid (in Volts).
		"""
		self.V_RF = V_RF      # RF amplitude
		self.f = f            # RF frequency (GHz)
		self.R_load = R_load
		self.V_px = V_px
		pass

	def load(self, file, N=2000, V_max=2e-3,interpN=2000,norm=[4e-3,5e-3],imax = 10e-9):
		"""
		Load an experimental dI/dV spectrum and extend it symmetrically.

		This ensures that when performing the Tien-Gordon convolution,
		the interpolation remains valid even when shifting the bias
		by multiple photon energies ±n*h*f/2e.

		Parameters:
		-----------
		file : str
			Path to Nanonis .dat file containing the bias and conductance.
		N : int
			Number of extra points added on each side of the spectrum.
		"""
		self.interpN = interpN
		self.imax = imax
		self.norm = norm
		x, y = self.fast_cond(file)  # Load the experimental data
		spac = x[0] - x[1]          # Voltage spacing between points
		# Create extended voltage array, symmetric around zero
		x_ext = np.arange(x[-1] - N * spac, x[0] + (N +1) * spac, spac)
		# Initialize extended conductance with 1 (background level)
		y_ext = np.zeros(y.shape[0] + 2 * N) + 1
		# Center experimental data around zero bias in the extended grid
		a = self.find_nearest(x_ext, 0)

		if a%10 == 0:
			y_ext[- a - len(y)//2 : a + len(y)//2] = y
			x_ext = np.arange(x[-1] - N * spac, x[0] + (N ) * spac, spac)
			print('attention!')
		else:
			y_ext[-1 - a - len(y)//2 : 1 + a + len(y)//2] = y

		# Store extended arrays for later use


		self.x_ext = x_ext
		self.y_ext = y_ext
		# Create an interpolation function (continuous dI/dV over bias)
		self.x_int = np.linspace(-V_max, V_max, interpN)  # Bias interpolation grid (for TG integration)
		self.V_max = V_max
		self.x_y_interp = sc.interpolate.interp1d(x_ext, y_ext)
		return x_ext,y_ext

	def load_curr(self, file, N=2000, V_max=2e-3,interpN=2000,norm=[4e-3,5e-3],imax = 10e-9):
		"""
		Load an experimental dI/dV spectrum and extend it symmetrically.

		This ensures that when performing the Tien-Gordon convolution,
		the interpolation remains valid even when shifting the bias
		by multiple photon energies ±n*h*f/2e.

		Parameters:
		-----------
		file : str
			Path to Nanonis .dat file containing the bias and conductance.
		N : int
			Number of extra points added on each side of the spectrum.
		"""
		self.V_max = V_max
		self.interpN = interpN
		self.norm = norm
		self.imax = imax
		x, y = self.fast_cur(file)  # Load the experimental data
		self.curr_max = y.max()
		spac = x[0] - x[1]          # Voltage spacing between points
		# Create extended voltage array, symmetric around zero
		x_ext = np.arange(x[-1] - N * spac, x[0] + (N +1) * spac, spac)
		# Initialize extended current
		y_ext = x_ext/(self.R)
		# Center experimental data around zero bias in the extended grid
		a = self.find_nearest(x_ext, 0)

		if a%10 == 0:
			y_ext[- a - len(y)//2 : 1+ a + len(y)//2] = np.flip(y)
			x_ext = np.arange(x[-1] - N * spac, x[0] + (N +1) * spac, spac)
			# print('attention')
		else:
			y_ext[-1 - a - len(y)//2 : 1 + a + len(y)//2] = np.flip(y)

		# Store extended arrays for later use
		self.x_ext = x_ext
		self.y_ext = y_ext
		# Create an interpolation function (continuous dI/dV over bias)
		self.x_int = np.linspace(-V_max, V_max, interpN)  # Bias interpolation grid (for TG integration)
		self.x_y_interp = sc.interpolate.interp1d(x_ext, y_ext)
		return x_ext,y_ext		
	
	def load_VI(self, file, N=2000, V_max=2e-3,interpN=2000,norm=[4e-3,5e-3],imax = 10e-9): # to load from VI data

		self.interpN = interpN
		self.norm = norm
		self.imax = imax
		self.V_max = V_max
		
		  # Load the experimental data
		spectra = nanonis.biasSpectroscopy()
		spectra.load(file)
		spectra.current_cal()
		# spectra.normalizeRange_symm(self.norm)
		
		x = spectra.biasVI_f.to_numpy()
		y = 1/spectra.conductance.to_numpy()
		y = y/y[0]
		self.R = x[0]/y[0]
		self.curr_max = y.max()
		spac = x[4] - x[5]          # Voltage spacing between points
		# Create extended voltage array, symmetric around zero
		# x_ext = np.arange()
		x_ext = np.arange( -N * spac,  (N ) * spac, spac)
		# Initialize extended conductance with 1 (background level)
		y_ext = np.zeros(len(x_ext))+1
		print(len(x_ext),len(y_ext))
		# Center experimental data around zero bias in the extended grid
		a = self.find_nearest(x_ext, 0)

		if a%10 == 0:
			y_ext[- a - len(y)//2 : a + len(y)//2] = y
			print('attention!')
		else:
			y_ext[a - len(y)//2 : a + len(y)//2] = y

		# Store extended arrays for later use
		self.x_ext = x_ext
		self.y_ext = y_ext
		# Create an interpolation function (continuous dI/dV over bias)
		self.x_int = np.linspace(-V_max, V_max, interpN)  # Bias interpolation grid (for TG integration)
		print(len(x_ext),len(y_ext))
		self.x_y_interp = sc.interpolate.interp1d(x_ext, y_ext)
		return x_ext,y_ext		
	
	def TG_map(self):
		"""
		Compute the Tien-Gordon map (photon-assisted tunneling spectrum).

		Each pixel corresponds to the effective conductance at a given RF amplitude.
		The current (or dI/dV) is obtained by summing over all photon-assisted
		processes weighted by Bessel functions J_n^2(α), where
		α = eV_RF / (ħω).
		"""
		map = []
		# Photon energy in Volts: hf / (2e)
		en = const.h * self.f * 1e9 / (2 * const.e)
		# Sweep the RF amplitude up to V_RF
		V = np.linspace(0, self.V_RF, self.V_px)
		self.V_RF_arr = V

		for j in V:  # iterate over RF amplitudes
			y_rf = np.zeros(self.interpN)
			# Sum over photon sidebands using Bessel weighting
			for i in range(-200, 200):
				y_rf += sc.special.jv(i, j/en)**2 * self.x_y_interp(self.x_int + 2 * i * en)
			map.append(y_rf)
		
		self.map = np.array(map)  # Store the final TG map as a 2D array
		return np.array(map)
	
	def load_RFmap(self,fnames, N=2000, V_max=1e-3,interpN=2000,norm=[4e-3,5e-3],imax = 2-9): #laod an measured tien godon map (current and transform to VI to apply TG shapiro)
		self.interpN = interpN
		self.V_max = V_max
		self.imax = imax
		self.norm = norm
		map = []
		for f in fnames:
			x,y = self.load_curr(f, N=N, V_max=V_max,interpN=interpN,norm=norm,imax=imax)

			map.append(y)

		self.map = np.array(map)
		return map

	def fast_cond(self, file):
		"""
		Load and normalize an experimental Nanonis dI/dV spectrum.

		Returns bias and conductance arrays.
		"""
		spectra = nanonis.biasSpectroscopy()
		spectra.load(file)
		spectra.normalizeRange(self.norm)
		x = np.array(spectra.bias)
		y = np.array(spectra.conductance)
		return x, y
	
	def fast_cur(self, file):
		"""
		Load and normalize an experimental Nanonis dI/dV spectrum.

		Returns bias and conductance arrays.
		"""
		spectra = nanonis.biasSpectroscopy()
		spectra.load(file)
		x = np.array(spectra.bias)
		y = np.array(spectra.current)
		self.R = x[0]/y[0]
		return x, y


	def find_nearest(self, array, value):
		"""
		Return the index of the array element closest to a target value.
		"""
		idx = (np.abs(array - value)).argmin()
		return idx

	def extend(self, x, y, N):
		"""
		(Alternative version) Extend data arrays symmetrically by N points.
		"""
		spac = self.x[0] - self.x[1]
		x_ext = np.arange(x[-1] - N * spac, x[0] + (N + 1) * spac, spac)
		y_ext = np.zeros(y.shape[0] + 2 * N) + 1
		a = self.find_nearest(x_ext, 0)
		y_ext[-1 - a - len(y)//2 : 1 + a + len(y)//2] = y
		return x_ext, y_ext

	def plot(self,gradient=False):
		"""
		Display the Tien-Gordon map as an image.

		X-axis: bias voltage (mV)
		Y-axis: RF amplitude (arbitrary units)
		"""
		f, ax = plt.subplots(1)
		if gradient == False:
			im = ax.imshow(self.map, aspect='auto',extent=[self.V_max*1e3,-self.V_max*1e3,0,self.V_RF])
		if gradient == True:
			im = ax.imshow(np.gradient(self.map)[0], aspect='auto')
		uf.add_clim_sliders(f,ax,im)
		ax.set_xlabel('Bias (mV)')
		ax.set_ylabel('Power (arbitrary)')

	def plot_power(self,gradient=False):
		"""
		Display the Tien-Gordon map as an image.

		X-axis: bias voltage (mV)
		Y-axis: RF amplitude (arbitrary units)
		"""
		f, ax = plt.subplots(1)
		if gradient == False:
			# im = ax.imshow(self.map, aspect='auto',extent=[self.V_max*1e3,-self.V_max*1e3,0,self.V_RF])
			im = ax.pcolormesh(np.linspace(self.V_max*1e3,-self.V_max*1e3,self.interpN),np.flipud(self.V_RF_arr**2),self.map)
		if gradient == True:
			im = ax.imshow(np.gradient(self.map)[0], aspect='auto')
		uf.add_clim_sliders(f,ax,im)
		ax.set_xlabel('Bias (mV)')
		ax.set_ylabel('Power (arbitrary)')

	def TG_shapiro(self): # calculate the IV power dependence with tien gordon
		shapiro_map_fwd = []
		shapiro_map_bwd = []

		n=0
		for i in range(0,self.map.shape[0]):
			shapiro_map_fwd.append(self.calc_VI(self.x_int,self.map[n,:])[1])
			shapiro_map_bwd.append(self.calc_VI(self.x_int,self.map[n,:])[2])
			n+=1
		self.shapiro_map_fwd = shapiro_map_fwd
		self.shapiro_map_bwd = shapiro_map_bwd
		return shapiro_map_fwd,shapiro_map_bwd
	
	def iloadline(self,abias,ibias):
		return ibias-abias/self.R_load
	
	def calc_VI(self,bias,curr):

		didvn=spsg.savgol_filter(curr,15,2,deriv=1,delta=np.ediff1d(bias)[0])
		curr=spsg.savgol_filter(curr,15,2)

		imax=self.imax
		prec=1e-7
		npts=200

		ibiass=np.linspace(-imax,imax,npts)
		bias=bias[np.abs(curr)<imax] #cut according to current smaller than imax
		didvn=didvn[np.abs(curr)<imax]
		curr=curr[np.abs(curr)<imax]
		curri=spi.interp1d(bias,curr,bounds_error=False,fill_value="extrapolate")
		abias=np.arange(np.min(bias),np.max(bias),prec) #voltage range at the given precision
		abias=np.arange(np.min(bias),np.max(bias),prec) #voltage range at the given precision
		acurr=curri(abias) #the interpolated current at the given precision
		voltsfwd=[]
		voltsbwd=[]
		for n,ibias in enumerate(ibiass): #the imposed bias currents
			ill=self.iloadline(abias,ibias) #current at the intended ibias for the range of resistor voltages
			ncross=np.argwhere(np.diff(np.sign(ill - acurr))).flatten() #index where the ll crosses the interpolated current
			ucross=abias[ncross]
			icross=acurr[ncross]
			voltsfwd.append(np.min(ucross))
			voltsbwd.append(np.max(ucross))
			
		voltsfwd=np.array(voltsfwd)
		voltsbwd=np.array(voltsbwd)
		return ibiass*1e9,voltsfwd, voltsbwd,bias,curr
	
	def load_line_check(self,fname):
		bs=nanonis.biasSpectroscopy()
		bs.load(fname)

		bias=bs.data["Bias calc (V)"]
		curr=bs.data["Current (A)"]
		didvn=spsg.savgol_filter(curr,15,2,deriv=1,delta=np.ediff1d(bias)[0])
		curr=spsg.savgol_filter(curr,15,2)
		lix=bs.data["LI Demod 1 X (A)"]
		didv=lix/(float(bs.header["Lock-in>Amplitude"])*1.5) #Correction factor from comparison to didvn

		R=1e6
		imax=self.imax

		prec=1e-7
		npts=200
		dec=1
		f=5e9
		print(2*const.e*f/1e-9)
		amps=np.linspace(0,10e-9,100)
		kmax=10

		ibiass=np.linspace(-imax,imax,npts)

		bias=bias[np.abs(curr)<imax] #cut according to current smaller than imax
		didv=didv[np.abs(curr)<imax]
		didvn=didvn[np.abs(curr)<imax]
		curr=curr[np.abs(curr)<imax]

		curri=spi.interp1d(bias,curr,bounds_error=False,fill_value="extrapolate")
		abias=np.arange(np.min(bias),np.max(bias),prec) #voltage range at the given precision

		fig,axs=plt.subplots(2,2,constrained_layout=True)

		acurr=curri(abias) #the interpolated current at the given precision
		axs[0,0].plot(abias*1e3,acurr*1e9,"-")

		voltsfwd=[]
		voltsbwd=[]
		for n,ibias in enumerate(ibiass): #the imposed bias currents
			ill=self.iloadline(abias,ibias) #current at the intended ibias for the range of resistor voltages
			
			ncross=np.argwhere     (np.diff(np.sign(ill - acurr))).flatten() #index where the ll crosses the interpolated current
			ucross=abias[ncross]
			icross=acurr[ncross]
			
			voltsfwd.append(np.min(ucross))
			voltsbwd.append(np.max(ucross))
			
			if n%dec==0 and len(ncross)>1:
				axs[0,0].plot(abias*1e3,ill*1e9,c="k",lw=0.1)
				axs[0,0].plot(ucross*1e3,icross*1e9,".",c="r",ms=3)
		voltsfwd=np.array(voltsfwd)
		voltsbwd=np.array(voltsbwd)

		axs[0,1].plot(ibiass*1e9,voltsfwd*1e3,".-",label="fwd",ms=3)
		axs[0,1].plot(ibiass*1e9,voltsbwd*1e3,".-",label="bwd",ms=3)
		axs[1,1].plot(ibiass*1e9,np.gradient(voltsfwd,ibiass)*1e-6,label="fwd")
		axs[1,1].plot(ibiass*1e9,np.gradient(voltsbwd,ibiass)*1e-6,label="bwd")

		axs[1,0].plot(bias*1e3,didvn*1e6)
		axs[1,0].plot(bias*1e3,didv*1e6)

		axs[0,1].legend()
		axs[1,1].legend()

		axs[0,0].set_xlabel("Sample bias $V$ (mV)")
		axs[0,0].set_title("Measured I-V curve")
		axs[0,0].set_ylabel("Current $I$ (nA)")

		axs[1,0].set_xlabel("Sample bias $V$ (mV)")
		axs[1,0].set_ylabel("$dI/dV$ ($uS)")
		axs[1,0].set_title("Measured dI/dV curve")

		axs[0,1].set_xlabel("Bias current $V_0/R$ (nA)")
		axs[0,1].set_title("Calculated V-I curves")

		axs[1,1].set_xlabel("Bias current $V_0/R$ (nA)")
		axs[0,1].set_ylabel("Sample voltage $V$ (mV)")
		axs[1,1].set_ylabel('dV/dI (MOhms)')
		axs[1,1].set_title("Calculated dV/dI curves")

	def plot_shapiro(self,bwd_fwd = 'fwd'):
		n=0
		if bwd_fwd == 'fwd':
			i = self.shapiro_map_fwd
		elif bwd_fwd == 'bwd':
			i = self.shapiro_map_bwd
		f,ax = plt.subplots(1)
		grad = np.gradient(i)[0]
		im = ax.imshow(np.flipud(grad),aspect='auto',extent=[-15,15,0,0.8],vmin=grad.min()*0.5,vmax=grad.max()*0.2,interpolation='nearest')
		ax.set_xlabel('Current bias (nA)')
		ax.set_ylabel('V_RF (mV)')
		ax.set_xlim(-self.imax*1e9,self.imax*1e9)
		ax.set_ylim(0,0.4)
		uf.add_clim_sliders(f,ax,im)
		pass