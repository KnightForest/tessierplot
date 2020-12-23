
import numpy as np
from scipy import signal
import scipy.constants as sc
import re
import collections
import matplotlib.colors as mplc
import matplotlib.pyplot as plt

REGEX_STYLE_WITH_PARAMS = re.compile('(.+)\((.+)\)')
REGEX_VARVALPAIR = re.compile('(\w+)=(.*)')

def nonzeromin(x):
	'''
	Get the smallest non-zero value from an array
	Also works recursively for multi-dimensional arrays
	Returns None if no non-zero values are found
	'''
	x = np.array(x)
	nzm = None
	if len(x.shape) > 1:
		for i in range(x.shape[0]):
			xnow = nonzeromin(x[i])
			if xnow != 0 and xnow is not None and (nzm is None or nzm > xnow):
				nzm = xnow
	else:
		for i in range(x.shape[0]):
			if x[i] != 0 and (nzm is None or nzm > x[i]):
				nzm = x[i]
	return nzm

def helper_deinterlace(w):
	w['deinterXXodd'] = w['XX'][1::2,1:] #take every other column in a sweepback measurement, offset 1
	w['deinterXXeven'] = w['XX'][::2,:] #offset 0

def helper_deinterlace0(w):
	w['XX'] = w['XX'][::2,:] #take even column in a sweepback measurement

def helper_deinterlace1(w):
	w['XX'] = w['XX'][1::2,1:] #take odd column in a sweepback measurement

def helper_mov_avg(w):
	(m, n) = (int(w['mov_avg_m']), int(w['mov_avg_n']))     # The shape of the window array
	
	data = w['XX']
	if data.ndim == 1:
		win = np.ones((n,))
		w['XX'] = moving_average_1d(w['XX'], win)
	else:
		win = np.ones((m, n))
		w['XX'] = moving_average_2d(w['XX'], win)

def helper_fixlabels(w):
	# List of hardcoded label replacements to make plots instantly look nicer. Will be different for every user/measurement though!

	# if not isinstance(w['ylabel'], np.ndarray):
	# 	ylabel = (w['ylabel'])
	# else:
	# 	ylabel = w['ylabel'][0]
	# 	yunit = w['ylabel'][1]
	# if not isinstance(w['xlabel'], np.ndarray):
	# 	xlabel = (w['xlabel'])
	# else:
	# 	xlabel = w['xlabel'][0]
	# 	xunit = w['xlabel'][1]

	xlabel = w['xlabel']
	xunit = w['xunit']
	ylabel = w['ylabel']
	yunit = w['yunit']
	cbar_q = (w['cbar_quantity'])
	cbar_u = (w['cbar_unit'])
	if cbar_q.find('Current') != -1:
		print('found current')
		cbar_q = '$I_\mathrm{D}$'
		cbar_unit = 'nA'
	if cbar_q.find('Voltage') != -1:
		print('found voltage')
		cbar_q = '$V_\mathrm{SD}$'
		cbar_unit = 'mV'

	if xlabel.find('mK') != -1:
		xlabel = '$T$'
		xunit = 'mK'
	elif xlabel.find('K') != -1:
		xlabel = '$T$'
		xunit = 'K'
	
	if xlabel.find('BG') != -1:
		xlabel = '$V_\mathrm{BG}$'
		xunit = 'mV'
	if xlabel.find('oop') != -1:
		xlabel = r'$B_{\bot}$'
		xunit = 'mT'
	
	if xlabel.find('B_X') != -1 or xlabel.find('Bx') != -1 or xlabel == 'x_field':
		xlabel = '$B_\mathrm{X}$'
		xunit = 'T'
	if xlabel.find('B_Y') != -1 or xlabel.find('By') != -1 or xlabel == 'y_field':
		xlabel = '$B_\mathrm{Y}$'
		xunit = 'T'
	if xlabel.find('B_Z') != -1 or xlabel.find('Bz') != -1 or xlabel == 'z_field':
		xlabel = '$B_\mathrm{Z}$'
		xunit = 'T'
	if xlabel.find('Power') != -1:
		xlabel = 'Applied RF Power'
		xunit = 'dBm'
	if ylabel.find('{g')!= -1 or ylabel.find('{G')!= -1:
		gn = re.search(r'\d+', ylabel).group()
		ylabel = '$V_\mathrm{g'+gn+'}$'
		yunit = 'mV'
	elif yunit.find('mV') != -1:
		ylabel = '$V_\mathrm{SD}$'
		yunit = 'mV'
	if xlabel.find('{g')!= -1 or xlabel.find('{G')!= -1:
		gn = re.search(r'\d+', xlabel).group()
		xlabel = '$V_\mathrm{g'+gn+'}$'
		xunit = 'mV'

	if yunit.find('nA') != -1:
		ylabel = '$I_\mathrm{S}$'
		yunit = 'nA'

	if ylabel == 'S21_frequency_set':
		ylabel = 'S21 freq.'
		yunit = 'Hz'
	if xlabel == 'VNA_S21_frequency_set':
		xlabel = 'S21 freq.'
		xunit = 'Hz'

	if ylabel == 'VNA_S21_magnitude':
		ylabel = 'S21 magn.'
		yunit = 'arb.'
	if ylabel == 'VNA_S21_phase':
		ylabel = 'S21 phase'
		yunit = '$\phi$'

	if cbar_q == 'VNA_S21_magnitude':
		cbar_q = 'S21 magn.'
		cbar_u = 'arb.'
	if cbar_q == 'VNA_S21_phase':
		cbar_u = '$\phi$'

	if xlabel == 'S21 frequency':
		xlabel = 'S21 freq.'
		xunit = 'Hz'
	if ylabel == 'S21 frequency':
		ylabel = 'S21 freq.'
		yunit = 'Hz'

	if ylabel == 'S21 magnitude':
		ylabel = 'S21 magn.'
		yunit = 'arb.'
	if ylabel == 'S21 phase':
		ylabel = 'S21 phase'
		yunit = '$\phi$'

	if cbar_q == 'S21 magnitude':
		cbar_q = 'S21 magn.'
		cbar_u = 'arb.'
	if cbar_q == 'VNA_S21_phase':
		cbar_u = '$\phi$'

	#Right now units for x and y are discarded and are hardcoded in fixlabels in the label itself. This may not be the best approach
	#but it's also inconsistent in plot.py between 2d and 3d plots. Running this style now always destroys x and y units as determined by
	# the default method.
	w['ylabel'] = ylabel
	w['yunit'] = yunit
	w['xlabel'] = xlabel
	w['xunit'] = xunit
	w['cbar_quantity'] = cbar_q
	w['cbar_unit'] = cbar_u


def helper_changeaxis(w):
	print(w['ext'])
	newext = (float(w['changeaxis_xfactor'])*w['ext'][0]+w['changeaxis_xoffset'],
			float(w['changeaxis_xfactor'])*w['ext'][1]+w['changeaxis_xoffset'],
			float(w['changeaxis_yfactor'])*w['ext'][2]+w['changeaxis_yoffset'],
			float(w['changeaxis_yfactor'])*w['ext'][3]+w['changeaxis_yoffset'])
	w['ext'] = newext
	w['XX'] = w['XX']*float(w['changeaxis_datafactor'])
	if w['changeaxis_dataunit'] != None:
		w['cbar_unit'] = w['changeaxis_dataunit']
	if w['changeaxis_xunit'] != None:
		w['xunit'] = w['changeaxis_xunit']
	if w['changeaxis_yunit'] != None:
		w['yunit'] = w['changeaxis_yunit']

def helper_diff(w):
	XX = w['XX']
	Y = w['Y']
	X = w['X']
	cbar_q = w['cbar_quantity']
	cbar_u = w['cbar_unit']
	condquant = strtobool(w['diff_condquant'])
	gradient = strtobool(w['diff_gradient'])
	axis = int(w['diff_axis'])
	order = int(w['diff_order'])
	if order is not 1:
		condquant = False
	#Keep axis selection with 0 or 1 compatibility:
	if axis == 0 or XX.ndim==1:
		axis = -1
	if axis == 1:
		axis = -2
	#Compute conductance quantum
	cq = 2*sc.elementary_charge**2/sc.h
	XX_t = XX
	if axis==-1 or XX.ndim==1: #Diff and smooth on fast axis
		if XX.ndim ==1:
			if order == 0:
				pass 
			elif order == 1:
				w['cbar_quantity'] = '$\partial$' + cbar_q + '/$\partial$' + w['xlabel']
			else: 
				w['cbar_quantity'] = '$\partial^{}$'.format(order) + cbar_q + '/$\partial$' + w['xlabel'] + '$^{}$'.format(order)
			for i in range(0,order):
				if gradient:
					XX_t = np.gradient(XX_t,X)
				else:
					XX_t = np.append(np.diff(XX_t)/np.diff(X),np.nan)
		else:	
			if order == 0:
				pass 
			elif order == 1:
				w['cbar_quantity'] = '$\partial$' + cbar_q + '/$\partial$' + w['ylabel']
			else:
				w['cbar_quantity'] = '$\partial^{}$'.format(order) + cbar_q + '/$\partial$' + w['ylabel'] + '$^{}$'.format(order)
			for i in range(0,XX.shape[0]):
				for j in range(0,order):
					if gradient:
						XX_t[i,:] = np.gradient(XX_t[i,:],Y[i,:])	
					else:
						XX_t[i,:-1] = np.diff(XX_t[i,:])/np.diff(Y[i,:])			
		if cbar_u == 'nA' and order == 1: 
			if condquant == True:
				XX_t = (1e-6 / cq) * XX_t
				w['cbar_unit'] = r'2$e^2$/h'
			else:
				w['cbar_unit'] = r'$\mu$S'
		elif cbar_u == 'A' and order ==1:
			if condquant == True:
				XX_t = (1/cq) * XX_t
				w['cbar_unit'] = r'2$e^2$/h'
			else:
				w['cbar_unit'] = r'S'
		
		elif cbar_u == 'mV' and order == 1:
			if condquant == True:
				XX_t = 1e-6*cq * XX_t
				w['cbar_unit'] = r'h/2$e^2$'
			else:
				XX_t = XX_t
				w['cbar_unit'] = r'M$\Omega$'
		elif cbar_u == 'V' and order ==1:
			if condquant == True:
				XX_t = cq * XX_t
				w['cbar_unit'] = r'h/2$e^2$'
			else:
				w['cbar_unit'] = r'$\Omega$'
		else:
			w['cbar_unit'] = ''
	
	elif axis==-2: #Diff and smooth in slow axis
		if order == 0:
			pass 
		elif order == 1:
			w['cbar_quantity'] = '$\partial$' + cbar_q + '/$\partial$' + w['xlabel']
		else:
			w['cbar_quantity'] = '$\partial^{}$'.format(order) + cbar_q + '/$\partial$' + w['xlabel'] + '$^{}$'.format(order)
		for i in range(0,XX.shape[1]):
			for j in range(0,order):
				if gradient:
					XX_t[:,i] = np.gradient(XX_t[:,i],X[:,i])
				else:
					XX_t[:-1,i] = np.diff(XX_t[:,i])/np.diff(X[:,i])	
		if cbar_u == 'nA' and order == 1: 
			if condquant == True:
				XX_t = (1e-6 / cq) * XX_t
				w['cbar_unit'] = r'2e$^2$/h'
			else:
				w['cbar_unit'] = r'$\mu$S'
		elif cbar_u == 'A' and order == 1:
			if condquant == True:
				XX_t = (1/cq) * XX_t
				w['cbar_unit'] = r'2e$^2$/h'
			else:
				w['cbar_unit'] = r'S'
		
		elif cbar_u == 'mV' and order == 1:
			if condquant == True:
				XX_t = (1e-6/cq) / XX_t
				w['cbar_unit'] = r'h/2e$^2$'
			else:
				w['cbar_unit'] = r'M$\Omega$'
		elif cbar_u == 'V' and order == 1:
			if condquant == True:
				XX_t = cq / XX_t
				w['cbar_unit'] = r'h/2e$^2$'
			else:
				w['cbar_unit'] = r'$\Omega$'
		else:
			w['cbar_unit'] = ''
	w['XX']=XX_t

def helper_savgol(w):
	'''Perform Savitzky-Golay smoothing and get nth order derivative on slow or fast axis'''
	'''WARNING! differentiation only works for evenly spaced datapoints'''
	XX = w['XX']
	Y = w['Y']
	X = w['X']
	cbar_q = w['cbar_quantity'] 
	cbar_u = w['cbar_unit']
	condquant = strtobool(w['savgol_condquant']) # Use units of e^2/h if possible
	axis = int(w['savgol_axis']) # -1 means fast axis, -2 slow axis differentiation
	difforder = int(w['savgol_difforder']) # Order of the derivative
	samples = int(w['savgol_samples']) # Samples for savgol filter
	order = int(w['savgol_order']) # Order of savgol filter
	if order is not 1:
		condquant = False
	#Keep axis selection with 0 or 1 compatibility:
	if axis == 0 or XX.ndim==1:
		axis = -1
	if axis == 1:
		axis = -2
	#Compute conductance quantum
	cq = 2*sc.elementary_charge**2/sc.h

	XX_t = XX
	if axis==-1 or XX.ndim==1: #Diff and smooth on fast axis
		if XX.ndim ==1:
			if difforder == 0:
				pass 
			elif difforder == 1:
				w['cbar_quantity'] = '$\partial$' + cbar_q + '/$\partial$' + w['xlabel']
			else: 
				w['cbar_quantity'] = '$\partial^{}$'.format(difforder) + cbar_q + '/$\partial$' + w['xlabel'] + '$^{}$'.format(difforder)
			XX_t = signal.savgol_filter(XX, samples, order, deriv=difforder, delta=np.diff(X)[0], mode='constant',cval=np.nan)
		else:
			if difforder == 0:
				pass 
			elif difforder == 1:
				w['cbar_quantity'] = '$\partial$' + cbar_q + '/$\partial$' + w['ylabel']
			else:
				w['cbar_quantity'] = '$\partial^{}$'.format(difforder) + cbar_q + '/$\partial$' + w['ylabel'] + '$^{}$'.format(difforder)
			y = np.diff(Y)
			yd = y[0]
			print(samples,order,difforder,yd,np.diff(Y)[0],axis)
			XX_t = signal.savgol_filter(XX, samples, order, deriv=difforder, delta=np.diff(Y,axis=axis)[0,0], axis=axis, mode='constant',cval=np.nan)

		if cbar_u == 'nA': 
			if condquant == True:
				XX_t = 1e6 * cq * XX_t
				w['cbar_unit'] = r'2e$^2$/h'
			else:
				w['cbar_unit'] = r'$\mu$S'
		if cbar_u == 'A':
			if condquant == True:
				XX_t = cq * XX_t
				w['cbar_unit'] = r'2e$^2$/h'
			else:
				w['cbar_unit'] = r'S'
		
		if cbar_u == 'mV':
			if condquant == True:
				XX_t = (1e6/cq) * XX_t
				w['cbar_unit'] = r'h/2e$^2$'
			else:
				w['cbar_unit'] = r'M$\Omega$'
		if cbar_u == 'V':
			if condquant == True:
				XX_t = (1/cq) * XX_t
				w['cbar_unit'] = r'h/2e$^2$'
			else:
				w['cbar_unit'] = r'$\Omega$'

	elif axis==-2: #Diff and smooth in slow axis
		if difforder == 0:
				pass 
		elif difforder == 1:
			w['cbar_quantity'] = '$\partial$' + cbar_q + '/$\partial$' + w['xlabel']
		else:
			w['cbar_quantity'] = '$\partial^{}$'.format(difforder) + cbar_q + '/$\partial$' + w['xlabel'] + '$^{}$'.format(difforder)
		XX_t = signal.savgol_filter(XX, samples, order, deriv=difforder, delta=np.diff(X,axis=axis)[0,0], axis=axis, mode='constant',cval=np.nan)
		if cbar_u == 'nA': 
			if condquant == True:
				XX_t = 1e6 * cq * XX_t
				w['cbar_unit'] = r'2e$^2$/h'
			else:
				w['cbar_unit'] = r'$\mu$S'
		if cbar_u == 'A':
			if condquant == True:
				XX_t = cq * XX_t
				w['cbar_unit'] = r'2e$^2$/h'
			else:
				w['cbar_unit'] = r'S'
		
		if cbar_u == 'mV':
			if condquant == True:
				XX_t = (1e6/cq) * XX_t
				w['cbar_unit'] = r'h/2e$^2$'
			else:
				w['cbar_unit'] = r'M$\Omega$'
		if cbar_u == 'V':
			if condquant == True:
				XX_t = (1/cq) * XX_t
				w['cbar_unit'] = r'h/2e$^2$'
			else:
				w['cbar_unit'] = r'$\Omega$'
	w['XX']=XX_t
		
def helper_hardgap(w): #Needs equally spaced axes
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	gaprange = [-float(w['hardgap_gaprange']),float(w['hardgap_gaprange'])]
	outsidegaprange = [float(w['hardgap_outsidegapmin']),float(w['hardgap_outsidegapmax'])]
	alphafactor = float(w['hardgap_alphafactor'])
	gaplimneg = np.abs(yaxis-gaprange[0]).argmin()
	gaplimpos = np.abs(yaxis-gaprange[1]).argmin()
	outsidegaplimneg = np.abs(yaxis-outsidegaprange[0]).argmin()
	outsidegaplimpos = np.abs(yaxis-outsidegaprange[1]).argmin()
	print(gaplimneg, gaplimpos, outsidegaplimneg,outsidegaplimpos)
	alllens, gapconductance,outsidegapconductance,hardness = np.array([None]*xn),np.array([None]*xn),np.array([None]*xn),np.array([None]*xn)
	for i in range(0,xn):
		gapconductance[i] = np.nanmean([XX[i,gaplimneg:gaplimpos]])
		outsidegapconductance[i] = np.nanmean([XX[i,outsidegaplimneg:outsidegaplimpos]])
	hardness = abs(gapconductance)/abs(outsidegapconductance)
	win = np.ones((2,))
	hardness = moving_average_1d(hardness[:], win)
	gateshift = np.nanmean(outsidegaprange)*1e-3/alphafactor
	gateshiftsteps = abs(int(round(gateshift/abs(xaxis[0]-xaxis[1]))))
	outsidegapconductancecorr = outsidegapconductance[gateshiftsteps::]
	outsidegapconductancecorr = np.append(outsidegapconductancecorr,[np.nan] * gateshiftsteps)
	print('gateshift:',gateshift,'gateshiftsteps:',gateshiftsteps,'outsidegapconductancecorr:',len(outsidegapconductancecorr), 'alphafactor', alphafactor)
	hardnesscorr = abs(gapconductance)/abs(outsidegapconductancecorr)
	hardnesscorr = moving_average_1d(hardnesscorr[:], win)

	fig = plt.figure()
	plt.plot(xaxis,outsidegapconductance,xaxis,gapconductance)		
	fig = plt.figure()
	plt.plot(xaxis,hardness)
	plt.yscale('log')
	xlabel = w['xlabel'] + ' ('+ w['xunit'] + ')'
	ylabel = '$G_\mathrm{G}/G_\mathrm{O}$'
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#Plot styling specified for 20170403\121507~1/121507_BiasSpec_W6_1E09_NW6T_LowNoiseaftertryingtokillit_ BG_2300-8500mV.dat.gz
	w['buffer']={'labels': [xlabel,ylabel], 'data':[[gapconductance],[outsidegapconductance],[hardness],[outsidegapconductancecorr],[hardnesscorr]], 'xaxis':[xaxis]}

def helper_int(w):
	XX = w['XX']
	X = w['X']
	Y = w['Y']
	ccstrtwo = r'2e$^2$/h' 
	ccstr = r'e$^2$/h' 
	modifier = 1
	if ccstrtwo in w['cbar_unit']:
		modifier = 	 (2*sc.elementary_charge**2/sc.h)
	elif ccstr in w['cbar_unit']:
		modifier = (sc.elementary_charge**2/sc.h)
	if XX.ndim == 1:
		intarr = np.nancumsum(XX)*np.diff(X,prepend=X[0]-(X[1]-X[0]))
		w['XX'] = (intarr - intarr[int(len(intarr)/2)])*modifier
		dxunit = w['xunit']
	else:
		for i in range(0,XX.shape[0]):
			intarr = np.nancumsum(XX[i,:])*np.diff(Y[i,:],prepend=Y[i,0]-(Y[i,1]-Y[i,0]))
			XX[i,:] = intarr - np.nanmean(intarr)
		w['XX']=XX*modifier
		dxunit = w['yunit']
	if modifier is not 1 and dxunit == 'V':
		w['cbar_unit'] = 'A'
	elif modifier is not 1 and dxunit == 'A':
		w['cbar_unit'] = 'int(S dA)'
	elif w['cbar_unit'] == 'S' and dxunit == 'V':
		w['cbar_unit'] = 'A'
	elif w['cbar_unit'] == r'$\mu$S' and dxunit == 'mV':
		w['cbar_unit'] = 'nA'
	elif w['cbar_unit'] == r'$\Omega$' and dxunit == 'A':
		w['cbar_unit'] = 'V'
	elif w['cbar_unit'] == r'k$\Omega$' and dxunit == 'nA':
		w['cbar_unit'] = 'mV'
	elif w['cbar_unit'] == r'M$\Omega$' and dxunit == 'nA':
		w['cbar_unit'] = r'$\mu$V'
	else:
		w['cbar_unit'] = 'int(' + w['cbar_unit'] + ')/d' + w['yunit']
	w['cbar_quantity'] = ''

def helper_log(w):
	w['XX'] = np.log10(np.abs(w['XX']))
	w['cbar_trans'] = ['log$_{10}$','abs'] + w['cbar_trans']
	w['cbar_quantity'] = w['cbar_quantity']
	w['cbar_unit'] = w['cbar_unit']
	
def helper_logdb(w):
	w['XX'] = 20*np.log10(np.abs(w['XX']))
	if w['cbar_quantity'] == '':
		w['yunit'] = 'dB'
	else:
		w['cbar_trans'] = ['20log$_{10}$','dB'] + w['cbar_trans']
		w['cbar_quantity'] = w['cbar_quantity']
		w['cbar_unit'] = w['cbar_unit']

def helper_logy(w):
	w['XX'] = np.log10(np.abs(w['XX']))
	w['cbar_trans'] = ['log$_{10}$','abs'] + w['cbar_trans']
	w['cbar_quantity'] = w['cbar_quantity']
	w['cbar_unit'] = w['cbar_unit']

def helper_fancylog(w):

	'''
	Use a logarithmic normalising function for plotting.
	This might be incompatible with Fiddle. <- Can confirm
	'''
	w['XX'] = abs(w['XX'])
	(cmin, cmax) = (w['fancylog_cmin'], w['fancylog_cmax'])
	if type(cmin) is str:
		cmin = float(cmin)
	if type(cmax) is str:
		cmax = float(cmax)
	if cmin is None:
		cmin = w['XX'].min()
		if cmin == 0:
			cmin = 0.1 * nonzeromin(w['XX'])
	if cmax is None:
		cmax = w['XX'].max()
	w['imshow_norm'] = mplc.LogNorm(vmin=cmin, vmax=cmax)

def helper_normal(w):
	pass
	
def helper_movingmeansubtract(w):  #Needs equally spaced axes
	XX = w['XX']
	xn, yn = XX.shape
	meanarray = np.zeros(xn)
	for i in range(0,xn):
		meanarray[i] = np.nanmean(XX[i][:])
	#print meanarray.shape
	win=int(w['movingmeansubtract_window'])
	print(win)
	padleft = int(round((win-1+0.0001)/2))
	padright = int(np.floor((win-1)/2))
	valleft = meanarray[0]
	valright = meanarray[-1]
	#print padleft,padright,valleft,valright
	meanarray = np.lib.pad(meanarray,(padleft,padright), 'constant', constant_values=(valleft,valright))
	#print meanarray.shape
	window = np.ones(win)
	window /= window.sum()
	if type(meanarray).__name__ not in ['ndarray', 'MaskedArray']:
		meanarray = np.asarray(meanarray)
	meanarray = signal.convolve(meanarray, window, mode='valid')
	#print meanarray.shape
	for i in range(0,len(meanarray)):
		XX[i] = XX[i]-meanarray[i]
	#fig = plt.figure()
	#plt.plot(meanarray)
	w['XX'] = XX
	
def helper_meansubtract(w):  #Needs equally spaced axes
	offset = np.nanmean(w['XX'])
	print('Subtracted mean:' + str(offset))
	w['XX'] = w['XX']-offset

def helper_deleteouterdatapoints(w):  #Needs equally spaced axes
	n = int((w['deleteouterdatapoints_n']))
	XX = w['XX']
	xn, yn = XX.shape
	newylim1 = w['ext'][2]+n*w['ystep']
	newylim2 = w['ext'][3]-n*w['ystep']
	print('n to be deleted:' + str(n))
	XX_new = np.zeros(shape=(xn,(yn-2*n)))
	for i in range(0,xn):
		y1 = n
		y2 = yn-n
		XX_new[i,:] = XX[i,y1:y2]
	w['ext'] = [w['ext'][0],w['ext'][1],newylim1,newylim2]
	w['XX'] = XX_new

def helper_offsetslopesubtract(w):  #Needs equally spaced axes
	offset,slope = (w['offsetslopesubtract_offset']),(w['offsetslopesubtract_slope'])
	#print offset
	xaxis = np.linspace(w['ext'][1],w['ext'][2],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	ymatrix = np.repeat([yaxis],w['XX'].shape[0], axis = 0)
	w['XX'] = w['XX']-(ymatrix*slope)-offset

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def helper_rshunt(w): #Needs equally spaced axes
	import math
	import numpy.ma as ma
	XX = w['XX']#+1e3/float(w['vbiascorrector_seriesr'])
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	#voffset = w['vbiascorrector_voffset'] # Voltage offset
	shuntr = w['rshunt_r'] # Series resistance
	ycorrected = np.zeros(shape=(xn,yn))
	gridresolutionfactor = w['rshunt_gridresolutionfactor'] # Example: Factor of 2 doubles the number of y datapoints for non-linear interpolation
	for i in range(0,xn):
		#print((XX[i,:]/yaxis))
		rtot = (XX[i,:]/yaxis)
		#print(rtot)
		#print(XX[i,:])
		try:
			rj = (1/((1/rtot)-(1/shuntr)))
		except:
			rj = 0 
		ij = XX[i,:]/rj
		#print(rj)
		#print(ij)
		ycorrected[i,:] = ij
	print(np.nanmin(ycorrected))
	ylimitneg,ylimitpos = (np.nanmin(ycorrected)), (np.nanmax(ycorrected))
	print(ylimitneg,ylimitpos)
	gridyaxis = np.linspace(ylimitneg,ylimitpos,int(yn*gridresolutionfactor))
	gridxaxis = xaxis
	XX_new = np.zeros(shape=(xn,len(gridyaxis)))
	if strtobool(w['rshunt_didv']) == True: # Calls didv within helper_vbiascorrector before interpolation to prevent artefacts.
		w['didv_condquant']=False
		helper_didv(w)
		XX = w['XX']
		for i in range(0,xn):
			testarr = ycorrected[i,:]
			nans, x= nan_helper(testarr)
			testarr[nans]= np.interp(x(nans), x(~nans), testarr[~nans])
			XX_new[i,:] = np.interp(gridyaxis,testarr[:-1],XX[i,:], left=np.nan, right=np.nan)
	else:
		for i in range(0,xn):
			testarr = ycorrected[i,:]
			nans, x= nan_helper(testarr)
			testarr[nans]= np.interp(x(nans), x(~nans), testarr[~nans])
			XX_new[i,:] = np.interp(gridyaxis,testarr,XX[i,:], left=np.nan, right=np.nan)
			
	w['XX'] = XX_new
	w['ystep'] = (ylimitpos-ylimitneg)/len(gridyaxis) #wrap ystep for analysis
	#print w['ystep']
	w['ext'] = [w['ext'][0],w['ext'][1],ylimitneg,ylimitpos]	

def helper_vbiascorrector(w): #Needs equally spaced axes
	import math
	import numpy.ma as ma
	XX = w['XX']#+1e3/float(w['vbiascorrector_seriesr'])
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	voffset = w['vbiascorrector_voffset'] # Voltage offset
	seriesr = w['vbiascorrector_seriesr'] # Series resistance
	ycorrected = np.zeros(shape=(xn,yn))
	gridresolutionfactor = w['vbiascorrector_gridresolutionfactor'] # Example: Factor of 2 doubles the number of y datapoints for non-linear interpolation
	for i in range(0,xn):
		ycorrected[i,:] = yaxis-voffset-XX[i,:]*seriesr*1e-3
	ylimitneg,ylimitpos = np.nanmin(ycorrected), np.nanmax(ycorrected)
	gridyaxis = np.linspace(ylimitneg,ylimitpos,int(yn*gridresolutionfactor))
	gridxaxis = xaxis
	XX_new = np.zeros(shape=(xn,len(gridyaxis)))
	if strtobool(w['vbiascorrector_didv']) == True: # Calls didv within helper_vbiascorrector before interpolation to prevent artefacts.
		helper_didv(w)
		XX = w['XX']
		#print XX.shape, w['XX'].shape
		for i in range(0,xn):
			XX_new[i,:] = np.interp(gridyaxis,ycorrected[i,:-1],XX[i,:], left=np.nan, right=np.nan)		#XX = w['XX']
	else:
		for i in range(0,xn):
			XX_new[i,:] = np.interp(gridyaxis,ycorrected[i,:],XX[i,:], left=np.nan, right=np.nan)
	w['XX'] = XX_new
	w['ystep'] = (ylimitpos-ylimitneg)/len(gridyaxis) #wrap ystep for analysis
	#print w['ystep']
	w['ext'] = [w['ext'][0],w['ext'][1],ylimitneg,ylimitpos]

def helper_ivreverserlockin(w):  #Needs equally spaced axes
	XX = w['XX']
	X = w['X']
	Y = w['Y']
	ccstrtwo = r'2e$^2$/h' 
	ccstr = r'e$^2$/h' 
	modifier = 1
	if ccstrtwo in w['cbar_unit']:
		modifier = 	 (2*sc.elementary_charge**2/sc.h)
	elif ccstr in w['cbar_unit']:
		modifier = (sc.elementary_charge**2/sc.h)
	if XX.ndim == 1:
		intarr = np.nancumsum(XX)*np.diff(X,prepend=X[0]-(X[1]-X[0]))
		XX = (intarr - intarr[int(len(intarr)/2)])*modifier
		dxunit = w['xunit']
	else:
		for i in range(0,XX.shape[0]):
			intarr = np.nancumsum(XX[i,:])*np.diff(Y[i,:],prepend=Y[i,0]-(Y[i,1]-Y[i,0]))
			a[i,:] = intarr - np.nanmean(intarr)
		XX=XX*modifier
		dxunit = w['yunit']
	if modifier is not 1 and dxunit == 'V':
		w['cbar_unit'] = 'A'
	elif modifier is not 1 and dxunit == 'A':
		w['cbar_unit'] = 'V'
	elif w['cbar_unit'] == 'S' and dxunit == 'V':
		w['cbar_unit'] = 'A'
	elif w['cbar_unit'] == r'$\mu$S' and dxunit == 'mV':
		w['cbar_unit'] = 'nA'
	elif w['cbar_unit'] == r'$\Omega$' and dxunit == 'A':
		w['cbar_unit'] = 'V'
	elif w['cbar_unit'] == r'k$\Omega$' and dxunit == 'nA':
		w['cbar_unit'] = 'mV'
	elif w['cbar_unit'] == r'M$\Omega$' and dxunit == 'nA':
		w['cbar_unit'] = r'$\mu$V'
	else:
		w['cbar_unit'] = 'int(' + w['cbar_unit'] + ')/d' + w['yunit']
	w['cbar_quantity'] = ''

	#Inverse I and V-bias measurements (works on both) by interpolating y-data on new homogeneous x-axis.
	# new versiong since matplotlibs griddata was deprecated :/
	from scipy.interpolate import griddata
	from scipy.interpolate import interp1d
	import math
	import numpy.ma as ma
	import numpy as np
	import numpy.matlib
	twodim = strtobool(w['ivreverser_twodim'])
	method = w['ivreverser_interpmethod']
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	ycorrected = np.zeros(shape=(xn,yn))
	gridresolutionfactor = int(w['ivreverser_gridresolutionfactor']) # Example: Factor of 2 doubles the number of y datapoints for non-linear interpolation
	
	for i in range(0,xn):
		ycorrected[i,:] = XX[i,:] #y-axis becomes data axis
		XX[i,:] = yaxis #data axis becomes y-axis (replace with repmat)
	ylimitneg,ylimitpos = (np.nanmin(ycorrected*10))/10, (np.nanmax(ycorrected*10))/10
	#print(ylimitneg,ylimitpos)
	grid_x, grid_y = np.mgrid[w['ext'][0]:w['ext'][1]:xn*1j, ylimitneg:ylimitpos:(yn*gridresolutionfactor)*1j]
	grid_y_1d = np.linspace(ylimitneg,ylimitpos,(yn*gridresolutionfactor))
	gridxstep = np.abs(grid_x[1,0]-grid_x[0,0])
	gridystep = np.abs(grid_y[0,1]-grid_y[0,0])
	#gridxstep,gridystep=1,1
	#print(gridxstep,gridystep)
	grid_x /= gridxstep
	grid_y /= gridystep 
	xrep = np.ravel(np.matlib.repmat(xaxis,yn,1),'F')
	points = np.transpose(np.vstack([xrep/gridxstep,np.ravel(ycorrected)/gridystep]))
	zf = np.ravel(XX)
	indexnonans=np.invert(np.isnan(points[:,0]))*np.invert(np.isnan(points[:,1]))*np.invert(np.isnan(zf))
	XX_new = np.zeros(shape=(xn,len(grid_y_1d)))
	# Calculate very tiny value relative to smallest value found in dataset to create a 
	# monotonous increase in the interpolated values. This prevents cubic interpolation 
	# from crashing since it cannot handle repeating values.
	minaddnp=np.abs(np.nanmin(XX)*1e-3)
	print(minaddnp)
	if twodim == True:
		print('2d')
		try:
			#XX_new = griddata(points, np.array(zf), (grid_x, grid_y), method='cubic')
			XX_new = griddata(np.stack((points[:,0][indexnonans],points[:,1][indexnonans]),axis=1), np.array(zf)[indexnonans], (grid_x, grid_y), method=method)
		except:
			XX_new = griddata(points, np.array(zf), (grid_x, grid_y), method=method)
			print('IVreverser {} interpolation failed, falling back to \'nearest\'.'.format(method))

	if twodim == False:
		print('1d')
		XX_new = np.zeros(shape=(xn,len(grid_y_1d)))
		try:
			for i in range(0,xn):
				indexnonans2 = np.nonzero(~np.isnan(ycorrected[i,:])) 
				ycn = ycorrected[i,(indexnonans2)][0]
				# Making sure ycn values are all unique by adding a random tiny value to each
				ycn = np.linspace(0,minaddnp*1e-6,len(ycn))+ycn
				yn = yaxis[indexnonans2]
				f = interp1d(ycn,yn, kind=method, bounds_error=False, fill_value=np.nan)
				XX_new[i,:] = f(grid_y_1d)
		except:
			print('IVreverser {} interpolation failed, falling back to \'nearest\'.'.format(method))
			for i in range(0,xn):
				f = interp1d(ycorrected[i,:],yaxis, kind='nearest', bounds_error=False, fill_value=np.nan)
				XX_new[i,:] = f(grid_y_1d)
	w['X'] = grid_x
	w['Y'] = grid_y
	w['XX'] = XX_new
	w['ext'] = (w['ext'][0],w['ext'][1],ylimitneg,ylimitpos)
	print(ylimitpos-ylimitneg,len(grid_y_1d)-1)
	w['ystep'] = np.abs(ylimitpos-ylimitneg)/(len(grid_y_1d)-1)
	print('new ystep:'+ str(w['ystep']))
	w['ext'] = (w['ext'][0],w['ext'][1],ylimitneg,ylimitpos)
	if w['yunit'].find('nA') != -1:
		print('I sourced detected')
		w['ylabel'] = '$V_\mathrm{SD}$'
		w['yunit'] = 'mV'
		w['cbar_quantity'] = '$I_\mathrm{S}$'
		w['cbar_unit'] = 'nA'
	elif w['yunit'].find('mV') != -1:
		print('V sourced detected')
		w['ylabel'] = '$I_\mathrm{D}$'
		w['yunit'] = 'nA'
		w['cbar_quantity'] = '$V_\mathrm{SD}$'
		w['cbar_unit'] = 'mV'
	elif w['yunit'].find('A') != -1:
		print('I sourced detected')
		w['ylabel'] = '$V_\mathrm{SD}$'
		w['yunit'] = 'V'
		w['cbar_quantity'] = '$I_\mathrm{S}$'
		w['cbar_unit'] = 'A'
	elif w['yunit'].find('V') != -1:
		print('V sourced detected')
		w['ylabel'] = '$I_\mathrm{D}$'
		w['yunit'] = 'A'
		w['cbar_quantity'] = '$V_\mathrm{SD}$'
		w['cbar_unit'] = 'V'

def helper_ivreverser(w):  #Needs equally spaced axes
	#Inverse I and V-bias measurements (works on both) by interpolating y-data on new homogeneous x-axis.
	# new versiong since matplotlibs griddata was deprecated :/
	from scipy.interpolate import griddata
	from scipy.interpolate import interp1d
	import math
	import numpy.ma as ma
	import numpy as np
	import numpy.matlib
	twodim = strtobool(w['ivreverser_twodim'])
	method = w['ivreverser_interpmethod']
	XX = w['XX']#+1e3/float(w['vbiascorrector_seriesr'])
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	ycorrected = np.zeros(shape=(xn,yn))
	gridresolutionfactor = int(w['ivreverser_gridresolutionfactor']) # Example: Factor of 2 doubles the number of y datapoints for non-linear interpolation
	
	for i in range(0,xn):
		ycorrected[i,:] = XX[i,:] #y-axis becomes data axis
		XX[i,:] = yaxis #data axis becomes y-axis (replace with repmat)
	ylimitneg,ylimitpos = (np.nanmin(ycorrected*10))/10, (np.nanmax(ycorrected*10))/10
	#print(ylimitneg,ylimitpos)
	grid_x, grid_y = np.mgrid[w['ext'][0]:w['ext'][1]:xn*1j, ylimitneg:ylimitpos:(yn*gridresolutionfactor)*1j]
	grid_y_1d = np.linspace(ylimitneg,ylimitpos,(yn*gridresolutionfactor))
	gridxstep = np.abs(grid_x[1,0]-grid_x[0,0])
	gridystep = np.abs(grid_y[0,1]-grid_y[0,0])
	#gridxstep,gridystep=1,1
	#print(gridxstep,gridystep)
	grid_x /= gridxstep
	grid_y /= gridystep 
	xrep = np.ravel(np.matlib.repmat(xaxis,yn,1),'F')
	points = np.transpose(np.vstack([xrep/gridxstep,np.ravel(ycorrected)/gridystep]))
	zf = np.ravel(XX)
	indexnonans=np.invert(np.isnan(points[:,0]))*np.invert(np.isnan(points[:,1]))*np.invert(np.isnan(zf))
	XX_new = np.zeros(shape=(xn,len(grid_y_1d)))
	# Calculate very tiny value relative to smallest value found in dataset to create a 
	# monotonous increase in the interpolated values. This prevents cubic interpolation 
	# from crashing since it cannot handle repeating values.
	minaddnp=np.abs(np.nanmin(XX)*1e-3)
	print(minaddnp)
	if twodim == True:
		print('2d')
		try:
			#XX_new = griddata(points, np.array(zf), (grid_x, grid_y), method='cubic')
			XX_new = griddata(np.stack((points[:,0][indexnonans],points[:,1][indexnonans]),axis=1), np.array(zf)[indexnonans], (grid_x, grid_y), method=method)
		except:
			XX_new = griddata(points, np.array(zf), (grid_x, grid_y), method=method)
			print('IVreverser {} interpolation failed, falling back to \'nearest\'.'.format(method))

	if twodim == False:
		print('1d')
		XX_new = np.zeros(shape=(xn,len(grid_y_1d)))
		try:
			for i in range(0,xn):
				indexnonans2 = np.nonzero(~np.isnan(ycorrected[i,:])) 
				ycn = ycorrected[i,(indexnonans2)][0]
				# Making sure ycn values are all unique by adding a random tiny value to each
				ycn = np.linspace(0,minaddnp*1e-6,len(ycn))+ycn
				yn = yaxis[indexnonans2]
				f = interp1d(ycn,yn, kind=method, bounds_error=False, fill_value=np.nan)
				XX_new[i,:] = f(grid_y_1d)
		except:
			print('IVreverser {} interpolation failed, falling back to \'nearest\'.'.format(method))
			for i in range(0,xn):
				f = interp1d(ycorrected[i,:],yaxis, kind='nearest', bounds_error=False, fill_value=np.nan)
				XX_new[i,:] = f(grid_y_1d)
	w['X'] = grid_x
	w['Y'] = grid_y
	w['XX'] = XX_new
	w['ext'] = (w['ext'][0],w['ext'][1],ylimitneg,ylimitpos)
	print(ylimitpos-ylimitneg,len(grid_y_1d)-1)
	w['ystep'] = np.abs(ylimitpos-ylimitneg)/(len(grid_y_1d)-1)
	print('new ystep:'+ str(w['ystep']))
	w['ext'] = (w['ext'][0],w['ext'][1],ylimitneg,ylimitpos)
	if w['yunit'].find('nA') != -1:
		print('I sourced detected')
		w['ylabel'] = '$V_\mathrm{SD}$'
		w['yunit'] = 'mV'
		w['cbar_quantity'] = '$I_\mathrm{S}$'
		w['cbar_unit'] = 'nA'
	elif w['yunit'].find('mV') != -1:
		print('V sourced detected')
		w['ylabel'] = '$I_\mathrm{D}$'
		w['yunit'] = 'nA'
		w['cbar_quantity'] = '$V_\mathrm{SD}$'
		w['cbar_unit'] = 'mV'
	elif w['yunit'].find('A') != -1:
		print('I sourced detected')
		w['ylabel'] = '$V_\mathrm{SD}$'
		w['yunit'] = 'V'
		w['cbar_quantity'] = '$I_\mathrm{S}$'
		w['cbar_unit'] = 'A'
	elif w['yunit'].find('V') != -1:
		print('V sourced detected')
		w['ylabel'] = '$I_\mathrm{D}$'
		w['yunit'] = 'A'
		w['cbar_quantity'] = '$V_\mathrm{SD}$'
		w['cbar_unit'] = 'V'

def helper_excesscurrent(w):  #Needs equally spaced axes
	#Designed for I-bias. Calculate excess current by performing linear fit at high bias and calculate the zero-crossing of the x-axis
	XX = w['XX']
	xn, yn = XX.shape
	limitfactor = w['excesscurrent_rangefactor'] #Percentual range of y-axis to use in polyfit. 
	#0.1 means that the top and bottom 10% are used (20% of total plot)
	limitfactor = limitfactor
	datacutoff = int(w['excesscurrent_datacutoff']) #Number of datapoints to discard at the beginning/end of the sweep
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	pospoly,negpoly,excesscurrpos,excesscurrneg = [None]*xn,[None]*xn,[None]*xn,[None]*xn
	fig = plt.figure()
	dataarray = np.zeros((xn,3))
	for i in range(0,xn):
		pospoly[i] = np.polyfit(yaxis[int(yn-(limitfactor*yn)):yn-datacutoff], XX[i,int(yn-(limitfactor*yn)):yn-datacutoff],1)
		negpoly[i] = np.polyfit(yaxis[datacutoff:int(limitfactor*yn)], XX[i,datacutoff:int(limitfactor*yn)],1)
		excesscurrpos[i] = -pospoly[i][1]/pospoly[i][0]
		excesscurrneg[i] = -negpoly[i][1]/negpoly[i][0]
		dataarray[i,2] = pospoly[i][0] #Rn
	dataarray[:,0] = excesscurrpos
	dataarray[:,1] = excesscurrneg
	xlabel = w['xlabel'] + ' ('+ w['xunit'] + ')'
	ylabel = w['ylabel'] + ' ('+ w['yunit'] + ')'
	plt.plot(xaxis,excesscurrpos,xaxis,excesscurrneg)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	w['buffer']={'labels': [xlabel,ylabel], 'data':[dataarray], 'xaxis':[xaxis], 'measdata':[XX]} #wrapping for analaysis
	
def helper_linecut(w):  #Needs equally spaced axes
	#Make a linecut on specified axis and value.
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	linecutvalue = w['linecut_value']
	if type(w['linecut_value']) is str: #If multiple linecuts are given separated by '_', unpack.
		linecutvalue = w['linecut_value'].split('!')
	else: 
		linecutvalue = [w['linecut_value']]
	print(linecutvalue)
	linecutvalue = [float(i) for i in linecutvalue]
	print(linecutvalue)
	axis = w['linecut_axis'] #Specified axis either 'x' or 'y'
	fig = plt.figure()
	if axis == 'x':
		dataarray = np.zeros((len(linecutvalue),yn))
		for val in range(len(linecutvalue)):
			xindex = np.abs(xaxis - (linecutvalue[val])).argmin()
			datavals = XX[xindex,:]
			plt.plot(yaxis,datavals)
			dataarray[val,:] = datavals
		xlabel = w['ylabel'] + ' ('+ w['yunit'] + ')'
		ylabel = w['cbar_quantity'] + ' ('+ w['cbar_unit'] + ')'
		xaxis = yaxis
	if axis == 'y':
		dataarray = np.zeros((len(linecutvalue),xn))
		for val in range(len(linecutvalue)):
			yindex = np.abs(yaxis - (linecutvalue[val])).argmin()
			datavals = XX[:,yindex]
			plt.plot(xaxis,datavals)
			dataarray[val,:] = datavals
		xlabel = w['xlabel'] + ' ('+ w['xunit'] + ')'
		ylabel = w['cbar_quantity'] + ' ('+ w['cbar_unit'] + ')'
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	w['buffer']={'labels': [xlabel,ylabel], 'data':[dataarray], 'xaxis':[xaxis], 'vals':[linecutvalue]} #wrapping for further analysis


def helper_resistance(w):  #Needs equally spaced axes
	#Calculate resistance of one sweep. Pretty old, not sure if it still works.
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	linecutvalue = w['resistance_linecutvalue']
	dopolyfit = strtobool(w['resistance_dolinearfit'])
	polyfitregion = w['resistance_fitregion']
	ylinecut = np.abs(yaxis - linecutvalue).argmin()
	if dopolyfit:
		ylinecutpolypos = np.abs(yaxis - (linecutvalue+polyfitregion)).argmin()
		ylinecutpolyneg = np.abs(yaxis - (linecutvalue-polyfitregion)).argmin()
	poly,resistance,conductance,notused2 = [None]*xn,[None]*xn,[None]*xn,[None]*xn
	fig = plt.figure()
	for i in range(0,xn):
		if dopolyfit==True:
			poly[i] = np.polyfit(yaxis[ylinecutpolyneg:ylinecutpolypos], XX[i,ylinecutpolyneg:ylinecutpolypos],1)
			resistance[i] = poly[i][0]*1e3 #kOhm
			conductance[i] = 1/poly[i][0] #microSiemens
		else:
			resistance[i] = XX[i,ylinecut]/yaxis[ylinecut]*1e3 #kOhm
			conductance = yaxis[ylinecut]/XX[i,ylinecut] #microSiemens
	plt.plot(xaxis,resistance)
	w['buffer'] = [xaxis,resistance,conductance]

def helper_dbmtovolt(w):  #Needs equally spaced axes
	#Convert dmb to 'volt', rf-amplifier (type?) conversion table included. Interpolates linear to logarithmic power axis.
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])

	def linrfamp(xaxis): #Conversion to volt using rf-amplifier
		dbin = np.linspace(-20,12,(20+12+1))
		dbout = [9.8, 10.7, 11.7, 12.6, 13.6, 14.5, 15.5, 16.5, 17.4, 18.4, 19.3, 20.2, 21.1, 22, 22.8, 23.5, 24, 24.8, 25.3, 25.7, 26.1, 26.4, 26.6, 26.8, 27, 27.1, 27.2, 27.3, 27.4, 27.45, 27.5, 27.5, 27.5]
		dbfit = np.poly1d(np.polyfit(dbin,dbout,deg=4))
		vxaxis = (dbfit(xaxis))
		return vxaxis

	def dbmtovolt(xaxis, attenuation):
		vxaxis = np.sqrt(np.power(10, ((xaxis-30-attenuation)/10)))
		return vxaxis

	if strtobool(w['dbmtovolt_rfamp'])==True: # Creates new power axis using attenuation and optional rf-amp
		xaxislin = dbmtovolt(linrfamp(xaxis), w['dbmtovolt_attenuation'])
	else:
		xaxislin = dbmtovolt(xaxis, w['dbmtovolt_attenuation'])
		ycorrected = np.zeros(shape=(xn,yn))
	
	gridresolutionfactor = int(w['dbmtovolt_gridresolutionfactor']) #Factor multiplying original resolution for interpolation of new power axis.
	xlimitneg,xlimitpos = np.nanmin(xaxislin), np.nanmax(xaxislin)
	gridxaxis = np.linspace(xlimitneg,xlimitpos,int(yn*gridresolutionfactor))
	gridyaxis = yaxis
	XX_new = np.zeros(shape=(len(gridxaxis),len(gridyaxis)))
	for i in range(0,yn):
		XX_new[:,i] = np.interp(gridxaxis,xaxislin,XX[:,i], left=np.nan, right=np.nan)
	w['XX'] = XX_new
	w['ext'] = (xlimitneg,xlimitpos,w['ext'][2],w['ext'][3])
	w['xstep'] = (xlimitpos-xlimitneg)/len(gridxaxis)
	w['xlabel'] = '$V_\mathrm{rms}$ (V)'


def helper_shapiro(w):  #Needs equally spaced axes
	#Looks for expected voltage of Shapiro step as function of applied frequency and order and returns closest measured index.
	planck = 4.135668e-15
	electron = 1#1.60e-19
	rffreq = w['shapiro_rffreq'] #Freq in GHz
	nsteps = int(w['shapiro_nsteps']) #Up to which order
	millivolts = w['shapiro_millivolts']
	if millivolts:
		w['XX'] = w['XX']*2*electron/(planck*rffreq*1000) #Convert XX to energy
	else:
		w['XX'] = w['XX']*2*electron/(planck*rffreq) #Convert XX to energy
	w['cbar_quantity'] = '$V_\mathrm{SD}$'
	w['cbar_unit'] = '$hf/2e$'
	#print rffreq
	XX = w['XX']
	xn, yn = XX.shape
	#print yn,xn
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	#ic = np.zeros((3,xn))
	#indexes = np.zeros(2)
	steppos = np.zeros((nsteps,xn))
	stepneg = np.zeros((nsteps,xn))
	fig = plt.figure()
	for n in range(nsteps):
		for i in range(0,xn):
			fullarray = XX[i,:]
			#print fullarray
			sposindex = np.abs(XX[i,:] - (n+0.5)).argmin()
			snegindex = np.abs(XX[i,:] - (-(n+0.5))).argmin()
			steppos[n][i] = yaxis[sposindex]
			stepneg[n][i] = yaxis[snegindex]
		plt.plot(xaxis,steppos[n][:],xaxis,stepneg[n][:])
	xlabel = w['xlabel'] + ' ('+ w['xunit'] + ')'
	#ylabel = w['cbar_quantity'] + ' ('+ w['cbar_unit'] + ')'
	ylabel = w['ylabel'] + ' ('+ w['yunit'] + ')'
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#print steppos.shape
	shapiro = [steppos,stepneg]
	w['buffer'] = shapiro
	w['buffer']={'labels': [xlabel,ylabel], 'data':[shapiro], 'xaxis':[xaxis]}

def helper_histogram(w):  #Needs equally spaced axes
	#Make a histogram
	XX = w['XX']
	xn, yn = XX.shape
	#print yn,xn
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	histbins = 250
	histrange = [w['histogram_rangemin'],w['histogram_rangemax']] #Range to histogram
	histbins = int(w['histogram_bins']) #Nr. of bins
	#print histbins,histrange
	newvaxis = np.linspace(histrange[0],histrange[1],histbins)
	fullhist = np.zeros(shape=(xn,histbins))
	#print fullhist.shape
	for i in range(0,xn):
		hist = np.histogram(XX[i,:],histbins, range=(histrange[0],histrange[1]))	
		fullhist[i,:] = hist[0][:]*w['ystep']
	w['XX'] = fullhist
	w['cbar_quantity'] = '$I_\mathrm{bins}$'
	w['cbar_unit'] = 'nA'
	w['ylabel'] = '$V_\mathrm{SD}$ (hf/2e)'
	w['ext'] = [w['ext'][0],w['ext'][1],histrange[0],histrange[1]]

def helper_histogram2(w):  #Needs equally spaced axes
	#Make histogram using stats.binned_statistics. Not sure what the advantage was.
	from scipy import stats
	XX = w['XX']
	xn, yn = XX.shape
	#print yn,xn
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	histrange = [-4,4]
	histbins = 500
	binrange = np.linspace(histrange[0],histrange[1],yn)
	isteps = np.zeros(shape=(xn,histbins))
	print(yaxis)
	for i in range(0,xn):
		bindata = XX[i,:]
		xvalsmax = stats.binned_statistic(bindata,yaxis, 'max', bins=histbins, range=histrange)
		print(xvalsmax)
		xvalsmin = stats.binned_statistic(bindata,yaxis, 'min', bins=histbins, range=histrange)
		print(xvalsmin)
		isteps[i] = xvalsmax[0][:]-xvalsmin[0][:]
		print(isteps[i])
	print(isteps)
	w['XX'] = isteps
	w['ext'] = [w['ext'][0],w['ext'][1],histrange[0],histrange[1]]

def helper_minsubtract(w): #Subtract smallest value from data
	w['XX'] = w['XX'] - np.min(w['XX'])

def helper_factor(w): #Multiply data with a factor
	w['XX'] = w['XX']*float(w['factor_factor'])

def helper_abs(w): #ABSOLUTELY
	w['XX'] = np.abs(w['XX'])
	#w['cbar_trans'] = ['abs'] + w['cbar_trans']
	w['cbar_quantity'] = '|' +w['cbar_quantity']+'|'
	w['cbar_unit'] = w['cbar_unit']

def helper_flipaxes(w):
	w['XX'] = np.transpose( w['XX'])
	w['ext'] = (w['ext'][2],w['ext'][3],w['ext'][0],w['ext'][1])
	
def helper_flipyaxis(w):
	#remember, this is before the final rot90
	#therefore lr, instead of ud
	w['XX'] = np.fliplr( w['XX'])
	w['ext'] = (w['ext'][0],w['ext'][1],w['ext'][3],w['ext'][2])

def helper_flipxaxis(w):
	w['XX'] = np.flipud( w['XX'])
	w['ext'] = (w['ext'][1],w['ext'][0],w['ext'][2],w['ext'][3])

def helper_crosscorr(w):
	A = w['XX']
	first = (w['crosscorr_toFirstColumn'])
	B = A.copy()
	#x in terms of linetrace, (y in terms of 3d plot)
	x = np.linspace(w['ext'][2],w['ext'][3],A.shape[1])
	x_org = x.copy()
	if w['crosscorr_peakmin'] is not None:
		if w['crosscorr_peakmin'] > w['crosscorr_peakmax']:
			peak = (x <= w['crosscorr_peakmin']) & (x >= w['crosscorr_peakmax'])
		else:
			peak = (x >= w['crosscorr_peakmin']) & (x <= w['crosscorr_peakmax'])

		B = B[:,peak]
		x = x[peak]
		
	offsets = np.array([])
	for i in range(B.shape[0]-1):
		#fit all peaks on B and calculate offset
		if first:
			column = B[0,:]
		else:
			column = B[i,:]
		
		next_column = B[i+1,:]
		
		offset = get_offset(x,column,next_column)
		offsets = np.append(offsets,offset)
		#modify A (and B for fun?S::S)
		A[i+1,:] = np.interp(x_org+offset,x_org,A[i+1,:])
		B[i+1,:] = np.interp(x+offset,x,B[i+1,:])
	
	w['offsets'] = 	offsets
	w['XX'] = A

def helper_deint_cross(w):  #Needs equally spaced axes
	A = w['XX']

	B = A.copy() 
	x = np.linspace(w['ext'][2],w['ext'][3],A.shape[1])
	
	#start at second column
	for i in (np.arange(B.shape[0]-1))[1::2]:
		#find offset of column to left and right
		column = B[i,:]
		
		previous_column = B[i-1,:]
		next_column = B[i+1,:]
		
		#average the offsets left and right
		offset = (get_offset(x,column,next_column)+get_offset(x,column,previous_column))/2.
		#modify A
		A[i,:] = np.interp(x-offset,x,A[i,:])
	w['XX'] = A

def helper_ic(w):  #Needs equally spaced axes
	#For meander + SC measurements: split file at zero bias and use 0 to finite sweep direction values for
	# both pos and neg bias. Next stick them together again.
    data = w['XX']
    helper_deinterlace(w)
    X0 = w['deinterXXodd']
    X1 = w['deinterXXeven']
    print(X0.shape, X1.shape)
    xn,yn = data.shape 
    # even/odd statements work on even deinterlacing (X1) since this is the largest
    if (yn % 2 == 0):
        print('y is even')
        cutb = int(yn/2+1)
        #print cutb
    else:
        print('y is uneven')
        cutb = int(yn/2)
        #print cutb
    #take the op of X0 (even) and check for even/uneven
    if(xn % 2 == 0):
        bottom = X1[:,cutb:]
    else:
        bottom = X1[1:,(cutb):]
    #take the bottom of X1 (oneven)
    top = X0[:,:(cutb-1)]
    w['XX']=np.hstack((top,bottom))

def helper_iretrap(w):  #Needs equally spaced axes
	#For meander + SC measurements: split file at zero bias and use finite to 0 sweep direction values (different 
	# from helper_ic) values for both pos and neg bias. Next stick them together again.
    data = w['XX']
    helper_deinterlace(w)
    X0 = w['deinterXXodd']
    X1 = w['deinterXXeven']
    xn,yn = data.shape
    ynr = int(np.round((yn/2)))
    #pak de onderkant van X0
    bottom = X0[:,ynr:]
    #pak de bovenkant van X1

    # even/odd statements work on even deinterlacing (X1) since this is the largest
    if (yn % 2 == 0):
        cutt = ynr-1
    else:
        cutt = ynr
    #take the op of X0 (even) and check for even/uneven
    if(xn % 2 == 0):
        top = X1[:,:cutt]
    else:
        top = X1[1:,:cutt]
    #top = X1[1:,:ynr]
    w['XX']=np.hstack((top,bottom))

def helper_icvsx(w):  #Needs equally spaced axes
	#Finds switching current or retrapping current by peak fitting. Preparation of data required by various other styles.
	# Correct use is complicated and requires lots of tuning (see comments). Styles to use before: 
	import sys
	from IPython.display import display
	import peakutils
	useonlythreshold = strtobool(w['icvsx_useonlythreshold']) #
	pixelnoiserange = int(w['icvsx_pixelnoiserange']) #size of significant noise in the superconducting state (value should be larger than noise period)
	peaktoplateauthreshold = w['icvsx_ppt'] #threshold ratio between normal state resistance and superconducting state
	stepoffset = w['icvsx_stepoffset'] #offset value in Is
	strictzero = strtobool(w['icvsx_strictzero']) #if True, absolute 0 is taken as a reference for the peaktoplateauthreshold
	limhigh = float(w['icvsx_plateaulim']) #max abs value of normal conductance (of lower, overrides value from data)
	limmin = float(w['icvsx_gapmax']) #'deadzone' for low bias
	print(useonlythreshold,pixelnoiserange,peaktoplateauthreshold,stepoffset,strictzero)
	
	# def reject_outliers(data, m = 2.):
		# d = np.abs(data - np.median(data))
		# mdev = np.median(d)
		# s = d/mdev if mdev else 0.
		# return data[s<m]
		
	#from peakutils.plot import plot as pplot
	from matplotlib import pyplot
	
	#Define axes
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	ic = np.zeros((3,xn))
	indexes = np.zeros(2)
	
	for i in range(0,xn):
		posarray = XX[i,int((yn/2)-1+stepoffset):0:-1] #get positive bias array
		negarray = XX[i,int((yn/2)+stepoffset):yn] #get negative bias array
		fullarray = XX[i,:] #fullarray

		posmax,negmax = np.nanmax(posarray),np.nanmax(negarray) #getting max values for both positive and negative bias
		posmin,negmin = np.nanmin(posarray),np.nanmin(negarray) #getting min values for both pos and neg halves
		poshigh,neghigh = np.nanmean(posarray[-20:-5]),np.nanmean(negarray[-20:-5]) #getting averaged 'high' values for both pos and neg halves, mor reliable
		
		#----
		# High and low values are used to determine the threshole peak_to_plateau_threshold for peak fitting 
		# (i.e. conductance in superconducting region vs conductance in normal state)
		#----
		
		# If high is higher than "limhigh = float(w['icvsx_plateaulim'])", value is overridden. This compensates for weird outliers
		if poshigh>limhigh:
			poshigh=limhigh
		if neghigh>limhigh:
			neghigh=limhigh
		
		# Check is reference should if absolute zero
		if strictzero == True:
			poslow,neglow = 0,0
		# If not, determine low values 
		else:
			poslow,neglow = np.nanmean((posarray[0:3]+negarray[0:3])/2),np.nanmean((posarray[0:3]+negarray[0:3])/2) # Average over a few datapoints around center of bias
			# Again check if low values should be overwritten with "limmin = float(w['icvsx_gapmax'])"
			if poslow>limmin:
				print('minlimfix')
				poslow=limmin
			if neglow>limmin:
				print('minlimfix')
				neglow=limmin

		# Determining the threshold in units of measured quantity 
		# (if poslow,neglow=0, its simply ppt*poshigh, i.e. the percentag of the high of the measured quantity)
		thresposabs = (poshigh - poslow)*peaktoplateauthreshold + poslow
		thresnegabs = (neghigh - neglow)*peaktoplateauthreshold + neglow
		
		# Thresholds calculated in fraction of maximum value (this is how peakfitting interprets threshold)
		threspos = (thresposabs-posmin)/(posmax-posmin)
		thresneg = (thresnegabs-negmin)/(negmax-negmin)
		#print(posmin,negmin,posmax,negmax)
		#print(poslow,neglow,poshigh,neghigh)
		#print(threspos,thresneg, thresposabs, thresnegabs)
	
		#-----------------------------
		# useonlythreshold bypasses peakfinding algorythm!
		#-----------------------------
		# Finds the first value in both posarray and negarray that is above thresposabs and thresnegabs.
		# This is the switch from superconducting to normal state.
		# (the thresholds in units of measured data), if not, there are no peaks to be found
		
		if useonlythreshold == True:
			try:
				indexnegnopeak = [x[0] for x in enumerate(negarray) if x[1] > thresnegabs]
				if len(indexnegnopeak) == 0:
					indexnegnopeak = np.array([0])
			except:
				e = sys.exc_info()[0]
				print(e)
				print('exceptneg')
			try:
				indexposnopeak = [x[0] for x in enumerate(posarray) if x[1] > thresposabs]
				if len(indexposnopeak) == 0:
					indexposnopeak = np.array([0])
			except:
				e = sys.exc_info()[0]
				print(e)
				print('exceptpos')
			finalindexneg = indexnegnopeak[0]
			finalindexpos = indexposnopeak[0]

		#-----------------------------
		# For more complicated or noisy data, the peakfitting method can be used (useonlythreshold == False)
		#-----------------------------
		else: 
			# Using peak fitting to determine all peaks in negative half of plot
			indexneg = peakutils.indexes(negarray,thres=thresneg, min_dist=1)
			
			# If no peak is found, set index to zero (ic=0)
			if indexneg.size == 0:
				indexneg = np.array([0])
			indexpos = peakutils.indexes(posarray,thres=threspos, min_dist=1)
			if indexpos.size == 0:
				indexpos = np.array([0])
			#--------
			# This code cycles through all found peak (up to 1000) and deletes them from the found peak indexlist 
			# if it thinks its just noise and does not correspond to a swithing event to the normal state.
			# This is determined by comparing the value a 'pixelnoiserange' number of datapoints away from the 
			# actual peak and checking if this value if not lower than five times the average of the whole array,
			# i.e. there should be a finite conductance after the peak is encountered.
			#--------
			firstpeakneg = indexneg[0]
			peaknoise = True
			counter = 0
			while peaknoise == True and counter < 1000:
				counter = counter + 1
				try:
					# Check if pek
					if True in (negarray[firstpeakneg:firstpeakneg+pixelnoiserange] < (-np.nanmean(negarray)/5)):
						indexneg = np.delete(indexneg,0)
						firstpeakneg = indexneg[0]
						finalindexneg = indexneg[0]
					else:
						peaknoise = False
						finalindexneg = indexneg[0]
				except:
					e = sys.exc_info()[0]
					print('negexcept')
					print(e)
					indexneg = np.array([0])
					if indexneg[0] + pixelnoiserange > len(negarray):
						finalindexneg = indexneg[0]#+int(yn/2)
					else:
						finalindexneg = 0
						print('Index ' + str(i) + 'neg not found')
					peaknoise = False
			firstpeakpos = indexpos[0]
			peaknoise = True
			counter = 0
			while peaknoise == True and counter < 1000:
				counter = counter + 1
				try:
					if True in (posarray[firstpeakpos:firstpeakpos+pixelnoiserange] < (-np.nanmean(posarray)/5)):
					#if True in (posarray[firstpeakpos:firstpeakpos+pixelnoiserange] < ( -posarray[firstpeakpos]/10)):
						indexpos = np.delete(indexpos,0)
						firstpeakpos = indexpos[0]
						finalindexpos = indexpos[0]
					else:
						peaknoise = False
						finalindexpos = indexpos[0]
				except:
					e = sys.exc_info()[0]
					print('posexcept')
					print(e)
					indexpos = np.array([0])
					if indexpos[0] + pixelnoiserange > len(posarray):
						finalindexpos = indexpos[0]+int(yn/2)
					else:
						finalindexpos = 0
						print('Index ' + str(i) + 'pos not found')
					peaknoise = False
		
		# After filtering out peaks that might be noise, the first real peak is selected
		if finalindexneg == 0:
			ic[0,i] = 0
		else:
			ic[0,i] = (yaxis[finalindexneg+int(yn/2)]) #neg one
		if finalindexpos == 0:
			ic[1,i] = 0
		else:
			ic[1,i] = (yaxis[finalindexpos+int(yn/2)]) #pos one

	fig = plt.figure()
	xlabel = w['xlabel'] + ' ('+ w['xunit'] + ')'
	ylabel = '$I_\mathrm{C}$ (nA)'
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.plot(xaxis,ic[0,:],xaxis,ic[1,:])#,ic[0,:],ic[2,:])
	w['buffer'] = ic
	w['buffer']={'labels': [xlabel,ylabel], 'data':[ic], 'xaxis':[xaxis]}
	return ic
	
def helper_massage(w):
	# func = w['massage_func']
	# func(w)
    func = w['massage_func']
    for f in func:
        f(w)

STYLE_FUNCS = {
	'abs': helper_abs,
	'changeaxis': helper_changeaxis,
	'crosscorr':helper_crosscorr,
	'dbmtovolt': helper_dbmtovolt,
	'deint_cross': helper_deint_cross,
	'deinterlace': helper_deinterlace,
	'deinterlace0': helper_deinterlace0,
	'deinterlace1': helper_deinterlace1,
	'deleteouterdatapoints': helper_deleteouterdatapoints,
	'diff': helper_diff,
	'excesscurrent': helper_excesscurrent,
	'factor': helper_factor,
	'fancylog': helper_fancylog,
	'fixlabels': helper_fixlabels,
	'flipaxes': helper_flipaxes,
	'flipxaxis': helper_flipxaxis,
	'flipyaxis': helper_flipyaxis,
	'hardgap': helper_hardgap,
	'histogram': helper_histogram,
	'ic': helper_ic,
	'icvsx': helper_icvsx,
	'int': helper_int,
	'iretrap': helper_iretrap,
	'ivreverser': helper_ivreverser,
	'linecut': helper_linecut,
	'log': helper_log,
	'logdb': helper_logdb,
	'massage': helper_massage,
	'meansubtract': helper_meansubtract,
	'minsubtract': helper_minsubtract,
	'mov_avg': helper_mov_avg,
	'movingmeansubtract': helper_movingmeansubtract,
	'normal': helper_normal,
	'offsetslopesubtract': helper_offsetslopesubtract,
	'resistance': helper_resistance,
	'rshunt': helper_rshunt,
	'savgol': helper_savgol,
	'shapiro': helper_shapiro,
	'vbiascorrector': helper_vbiascorrector,
	}

'''
Specification of styles with arguments
Format:
	{'<style_name>': {'<param_name>': <default_value>, 'param_order': order}}
Multiple styles can be specified, multiple parameters (name-defaultvalue pairs)
can be specified, and param_order decides the order in which they can be given
as non-keyword arguments.
'''
STYLE_SPECS = {
	'abs': {'param_order': []},
	'changeaxis': {'xfactor': 1, 'yfactor': 1, 'xoffset': 0, 'yoffset': 0,'datafactor': 1, 'dataunit': None, 'xunit': None, 'yunit': None, 'param_order': ['xfactor','yfactor','xoffset','yoffset','xunit','yunit', 'datafactor', 'dataunit']},
	'crosscorr': {'peakmin':None,'peakmax':None,'toFirstColumn':True,'param_order': ['peakmin','peakmax','toFirstColumn']},
	'dbmtovolt': {'rfamp': False, 'attenuation': 0, 'gridresolutionfactor': 2, 'param_order': ['rfamp','attenuation','gridresolutionfactor']},
	'deint_cross': {'param_order': []},
	'deinterlace': {'param_order': []},
	'deinterlace0': {'param_order': []},
	'deinterlace1': {'param_order': []},
	'deleteouterdatapoints': {'n':0,'param_order': ['n']},
	'diff': {'condquant': False, 'axis': -1, 'gradient': True, 'order': 1, 'param_order': ['condquant','axis','gradient','order']},
	'excesscurrent': {'datacutoff': 3, 'rangefactor': 0.15, 'plot': 0, 'plotval': 0,'param_order': ['datacutoff','rangefactor','plot','plotval']},
	'factor': {'factor':1,'param_order': ['factor']},
	'fancylog': {'cmin': None, 'cmax': None, 'param_order': ['cmin', 'cmax']},
	'fixlabels': {'param_order': []},
	'flipaxes': {'param_order': []},
	'flipyaxis': {'param_order': []},
	'flipxaxis': {'param_order': []},
	'hardgap': {'gaprange': 0.1, 'outsidegapmin': 0.5, 'outsidegapmax': 0.6, 'alphafactor': 1e9, 'param_order': ['gaprange','outsidegapmin','outsidegapmax','alphafactor']},
	'histogram':{'bins': 25, 'rangemin': -1, 'rangemax': 1, 'param_order': ['bins','rangemin','rangemax']},
	'ic': {'param_order': []},
	'icvsx': {'useonlythreshold': True, 'pixelnoiserange': 3, 'ppt': 0.5, 'stepoffset': 0, 'strictzero': True, 'plateaulim': 1e6,'gapmax': 1e6,'param_order': ['useonlythreshold','pixelnoiserange','ppt','stepoffset','strictzero','plateaulim','gapmax']},
	'int': {'param_order': []},
	'iretrap': {'param_order': []},
	'ivreverser':{'gridresolutionfactor': 2, 'twodim': False, 'interpmethod': 'cubic', 'param_order': ['gridresolutionfactor','twodim','interpmethod']},
	'linecut': {'linecutvalue': 1,'axis': None,'param_order': ['linecutvalue','axis']},
	'log': {'param_order': []},
	'logdb': {'param_order': []},
	'massage': {'param_order': []},
	'meansubtract': {'param_order': []},
	'minsubtract': {'param_order': []},
	'mov_avg': {'m': 1, 'n': 3, 'win': None, 'param_order': ['m', 'n', 'win']},
	'movingmeansubtract': {'window': 2,'param_order': ['window']},
	'normal': {'param_order': []},
	'offsetslopesubtract': {'slope': 0, 'offset': 0, 'param_order': ['slope', 'offset']},
	'resistance': {'linecutvalue': 0, 'dolinearfit': False, 'fitregion': 1, 'param_order': ['linecutvalue','dolinearfit','fitregion']},
	'rshunt': {'r':1e-10,'gridresolutionfactor': 2, 'didv': False,'param_order': ['r','gridresolutionfactor','didv']},
	'savgol': {'condquant': False, 'axis': -1, 'difforder':1, 'samples': 7, 'order': 3, 'param_order': ['condquant','axis','difforder','samples','order']},
	'sgtwodidv': {'samples': 21, 'order': 3, 'param_order': ['samples', 'order']},
	'shapiro': {'rffreq': 2.15e9, 'nsteps': 1, 'millivolts': 1, 'param_order': ['rffreq','nsteps','millivolts']},
	'ssidrive': {'param_order': []},
	'vbiascorrector':{'voffset': 0,'seriesr': 0, 'gridresolutionfactor': 1, 'didv':False, 'param_order': ['voffset','seriesr','gridresolutionfactor','didv']},
}

	#linecutvalue = w['linecut_value']
	#axis = w['linecut_axis']

	# useonlythreshold = 1
	# pixelnoiserange = 10
	# peaktoplateauthreshold = 0.5 # threshold
	# stepoffset = 0
	# strictzero = True
#Backward compatibility
styles = STYLE_FUNCS

def getEmptyWrap():
	'''Get empty wrap with default parameter values'''
	w = {'ext':(0,0,0,0), 
		'xstep':1,
		'ystep':1,
		'X': [],
		'Y': [],
		'XX': [], 
		'cbar_quantity': '',
		'cbar_unit': 'a.u.',
		'cbar_trans': [],
		'xlabel': [],
		'ylabel': [],
		'xunit': [],
		'yunit': [],
		'buffer': [],
		'imshow_norm': None}
	return w

def getPopulatedWrap(style=[]):
	'''
	Get wrap with populated values specified in the style list
	For example, if styles is:
		['deinterlace', 'mov_avg(n=5, 1)']
	This will add the following to the wrap:
		{'mov_avg_n': '5', 'mov_avg_m': '1'}
	'''
	w = getEmptyWrap()
	if style is None:
		return w
	elif type(style) is not list:
		style = list([style])
	for s in style:
		try:
			# Parse, process keyword arguments and collect non-kw arguments
			sregex = re.match(REGEX_STYLE_WITH_PARAMS, s)
			spar = []
			if sregex is not None:
				(s, sparamstr) = sregex.group(1,2)
				sparams = (
						sparamstr.replace(';',',').replace(':','=')
						.split(','))
				for i in range(len(sparams)):
					sparam = sparams[i].strip()
					spregex = re.match(REGEX_VARVALPAIR, sparam)
					if spregex is None:
						spar.append(sparam)
					else:
						val = spregex.group(2)
						#bool posing as string?
						if val.lower() == True:
							val = True
						elif val.lower() == False:
							val = False
						if type(val) is not bool:
							try:
								val = float(val)
							except ValueError:
								pass						
						w['{:s}_{:s}'.format(s, spregex.group(1))] = val
	
			# Process non-keyword arguments and default values
			(i, j) = (0, 0)
			pnames = STYLE_SPECS[s]['param_order']
			while i < len(pnames):
				key = '{:s}_{:s}'.format(s, pnames[i])
				if key not in w:
					if j < len(spar):
						w[key] = spar[j]
						j += 1
					else:
						w[key] = STYLE_SPECS[s][pnames[i]]
				i += 1
		except Exception as e:
			print('getPolulatedWrap(): Style {:s} does not exist ({:s})'.format(s, str(e)))
			print(e.__doc__)
			print(e.args)
			pass
	return w

def processStyle(style, wrap):
	for s in style:
		try:
#			print(s)
			func = STYLE_FUNCS[s.split('(')[0]]
			func(wrap)
		except Exception as e:
			print('processStyle(): Style {:s} does not exist ({:s})'.format(s, str(e)))
			print(e.__doc__)
			print(e.args)
			pass


def moving_average_2d(data, window):
    """Moving average on two-dimensional data.
    """
    # Makes sure that the window function is normalized.
    window /= window.sum()
    # Makes sure data array is a np array or masked array.
    if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = np.asarray(data)

    # The output array has the same dimensions as the input data
    # (mode='same') and symmetrical boundary conditions are assumed
    # (boundary='symm').
    return signal.convolve2d(data, window, mode='same', boundary='fill', fillvalue=np.nan)

def moving_average_1d(data, window):
    """Moving average on two-dimensional data.
    """
    # Makes sure that the window function is normalized.
    window /= window.sum()
    # Makes sure data array is a np array or masked array.
    if type(data).__name__ not in ['ndarray', 'MaskedArray']:
        data = np.asarray(data)

    # The output array has the same dimensions as the input data
    # (mode='same') and symmetrical boundary conditions are assumed
    # (boundary='symm').
    smoothed = signal.convolve(data, window, mode='same')
    smoothed[-(np.int(len(window)/2)):] = np.nan
    smoothed[0:np.int(len(window)/2)] = np.nan   
    return smoothed


def get_offset(x,y1,y2):
    corr = np.correlate(y1,y2,mode='same')

    #do a fit with a standard parabola for subpixel accuracy
    import lmfit
    from lmfit.models import ParabolicModel
    mod = ParabolicModel()

    def parabola(x,x0,a,b,c):
        return a*np.power((x-x0),2)+b*(x-x0)+c
    mod = lmfit.Model(parabola)

    _threshold = 0.7*max(corr)
    xcorr=(np.linspace(0,len(corr),len(corr)))
    xcorrfit=xcorr[corr >= _threshold]
    ycorrfit=corr[corr >= _threshold]

    
    p=lmfit.Parameters()
            #           (Name,  Value,  Vary,   Min,  Max,  Expr)

    p.add_many(        ('x0',      xcorrfit[ycorrfit==np.max(ycorrfit)][0],True,None,None,None),
                        ('a',      -1,True,None,1,None),
                       ('c',    1.,  True, None,None ,  None),
                        ('b',0, False,  None,None,None)
               )
    result = mod.fit(ycorrfit,params=p,x=xcorrfit)
#     print result.fit_report()
    # todo:if result poorly conditioned throw it out and make offset 0
    
    x0=result.best_values['x0']
    span = max(x)-min(x)
    #map back to real x values
    xmap= np.linspace(
				span/2, -span/2
				,len(xcorr)) 
    offset_intp = np.interp(x0,xcorr,xmap)
    return offset_intp

def strtobool(string):
	if string == 'True':
		string = 1
	else:
		string = 0
	return string