import numpy as np
from scipy import signal
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
	#w.deinterXXodd  = w.deinterXXodd
	#w.deinterXXeven = w.deinterXXeven

def helper_deinterlace0(w):
	w['XX'] = w['XX'][::2,:] #take even column in a sweepback measurement

def helper_deinterlace1(w):
	w['XX'] = w['XX'][1::2,1:] #take odd column in a sweepback measurement
	

def helper_mov_avg(w):
	(m, n) = (int(w['mov_avg_m']), int(w['mov_avg_n']))     # The shape of the window array
	
	data = w['XX']
	if data.ndim == 1:
		win = np.ones((m,))
		w['XX'] = moving_average_1d(w['XX'], win)
	else:
		win = np.ones((m, n))
		w['XX'] = moving_average_2d(w['XX'], win)
	#win = signal.kaiser(m,8.6,sym=False)

def helper_fixlabels(w):
	ylabel = (w['ylabel'])
	xlabel = (w['xlabel'])
	cbar_q = (w['cbar_quantity'])
	cbar_u = (w['cbar_unit'])
	#print(cbar_q, cbar_u)
	#print(ylabel, xlabel)
	if cbar_q.find('Current') != -1:
		print('found current')
		cbar_q = '$I_\mathrm{D}$'
		cbar_unit = '(nA)'
	if cbar_q.find('Voltage') != -1:
		print('found voltage')
		cbar_q = '$V_\mathrm{SD}$'
		cbar_unit = '(mV)'

	if xlabel.find('mK') != -1:
		xlabel = '$T$ (mK)'
	elif xlabel.find('K') != -1:
		xlabel = '$T$ (K)'
	
	if xlabel.find('BG') != -1:
		xlabel = '$V_\mathrm{BG}$ (mV)'
	if xlabel.find('oop') != -1:
		xlabel = r'$B_{\bot}$ (mT)'
	
	if xlabel.find('B_X') != -1 or xlabel.find('Bx') != -1 or xlabel == 'x_field':
		xlabel = '$B_\mathrm{X}$ (T)'
	if xlabel.find('B_Y') != -1 or xlabel.find('By') != -1 or xlabel == 'y_field':
		xlabel = '$B_\mathrm{Y}$ (T)'
	if xlabel.find('B_Z') != -1 or xlabel.find('Bz') != -1 or xlabel == 'z_field':
		xlabel = '$B_\mathrm{Z}$ (T)'
	if xlabel.find('Power') != -1:
		xlabel = 'Applied RF Power (dBm)'
	if ylabel.find('{g')!= -1 or ylabel.find('{G')!= -1:
		gn = re.search(r'\d+', ylabel).group()
		ylabel = '$V_\mathrm{g'+gn+'}$ (mV)'
	elif ylabel.find('mV') != -1:
		ylabel = '$V_\mathrm{SD}$ (mV)'
	if xlabel.find('{g')!= -1 or xlabel.find('{G')!= -1:
		gn = re.search(r'\d+', xlabel).group()
		xlabel = '$V_\mathrm{g'+gn+'}$ (mV)'

	if ylabel.find('nA') != -1:
		ylabel = '$I_\mathrm{S}$ (nA)'

	if ylabel == 'S21 frequency':
		ylabel = 'S21 freq. (Hz)'
	if xlabel == 'S21 frequency':
		xlabel = 'S21 freq. (Hz)'

	if cbar_q == 'S21 magnitude':
		cbar_q = 'S21 magn.'
		cbar_u = 'arb.'
	if cbar_q == 'S21 phase':
		cbar_u = '$\phi$'

	w['ylabel'] = ylabel
	w['xlabel'] = xlabel
	w['cbar_quantity'] = cbar_q
	w['cbar_unit'] = cbar_u
	#print(w)

def helper_changeaxis(w):
	print(w['ext'])
	newext = (float(w['changeaxis_xfactor'])*w['ext'][0],float(w['changeaxis_xfactor'])*w['ext'][1],float(w['changeaxis_yfactor'])*w['ext'][2],float(w['changeaxis_yfactor'])*w['ext'][3])
	w['ext'] = newext
	print(w['ext'])
	#print float(w['changeaxis_datafactor'])
	w['XX'] = w['XX']*float(w['changeaxis_datafactor'])
	if w['changeaxis_dataunit'] != None:
		w['cbar_unit'] = w['changeaxis_dataunit']
	if w['changeaxis_xunit'] != None:
		xsplit = w['xlabel'].rsplit('(')
		w['xlabel'] = xsplit[0] + '(' + w['changeaxis_xunit'] + ')'
	if w['changeaxis_yunit'] != None:
		ysplit = w['ylabel'].rsplit('(')
		w['ylabel'] = ysplit[0] + '(' + w['changeaxis_yunit'] + ')'

def helper_savgol(w):
	'''Perform Savitzky-Golay smoothing'''
	w['XX'] = signal.savgol_filter(
			w['XX'], int(w['savgol_samples']), int(w['savgol_order']))

def helper_didv(w):
	a=(w['XX'])
	ylabel = w['ylabel']
	xlabel = w['xlabel']
	cbar_q = w['cbar_quantity']
	cbar_u = w['cbar_unit']
	condquant = w['didv_condquant']

	if a.ndim == 1: #if 1D 
		w['XX'] = np.diff(w['XX'])
		w['XX'] = np.append(w['XX'],w['XX'][-1])
	else:
		w['XX'] = np.diff(w['XX'],axis=1)
	
	w['cbar_quantity'] = '$\partial$' + cbar_q + '/$\partial$' + ylabel.split(' ', 1)[0]
	
	if cbar_u == 'nA':
		#1 nA / 1 mV = 0.0129064037 conductance quanta
		if condquant==True:
			w['XX'] = w['XX'] / w['ystep'] * 0.0129064037
			w['cbar_unit'] = r'2$\mathrm{e}^2/\mathrm{h}$'
		else:
			w['XX'] = w['XX'] / w['ystep']
			w['cbar_unit'] = '$\mu$S'
	elif cbar_u == 'mV':
		#1 mV / 1 nA = 77.4809173 conductance resistance
		if condquant==True:
			w['XX'] = w['XX'] / w['ystep'] * 77.4809173
			w['cbar_unit'] = r'$\mathrm{h}/2\mathrm{e}^2$'
		else:
			w['XX'] = w['XX'] / w['ystep']
			w['cbar_unit'] = '$\mathrm{M}\Omega$'
	else:
		if condquant == True:
			w['XX'] = w['XX'] / w['ystep'] * 0.0129064037
		else:
			w['XX'] = w['XX'] / w['ystep']# * 0.0129064037
		w['cbar_unit'] = ''

def helper_hardgap(w):
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	gaprange = [-float(w['hardgap_gaprange']),float(w['hardgap_gaprange'])]
	outsidegaprange = [float(w['hardgap_outsidegapmin']),float(w['hardgap_outsidegapmax'])]
	gaplimneg = np.abs(yaxis-gaprange[0]).argmin()
	gaplimpos = np.abs(yaxis-gaprange[1]).argmin()
	outsidegaplimneg = np.abs(yaxis-outsidegaprange[0]).argmin()
	outsidegaplimpos = np.abs(yaxis-outsidegaprange[1]).argmin()
	print(gaplimneg, gaplimpos, outsidegaplimneg,outsidegaplimpos)
	alllens, gapconductance,outsidegapconductance,hardness = np.array([None]*xn),np.array([None]*xn),np.array([None]*xn),np.array([None]*xn)
	for i in range(0,xn):
		gapconductance[i] = np.mean([XX[i,gaplimneg:gaplimpos]])
		outsidegapconductance[i] = np.mean([XX[i,outsidegaplimneg:outsidegaplimpos]])
	hardness = abs(gapconductance)/abs(outsidegapconductance)
	win = np.ones((2,))
	hardness = moving_average_1d(hardness[:], win)
	fig = plt.figure()
	plt.plot(xaxis,outsidegapconductance,xaxis,gapconductance)#,xaxis,gapconductance)		
	fig = plt.figure()
	plt.plot(xaxis,hardness)
	plt.yscale('log')
	xlabel = w['xlabel']
	ylabel = '$G_\mathrm{G}/G_\mathrm{O}$'
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#Plot styling specified for 20170403\121507~1/121507_BiasSpec_W6_1E09_NW6T_LowNoiseaftertryingtokillit_ BG_2300-8500mV.dat.gz
	w['buffer']={'labels': [xlabel,ylabel], 'data':[[gapconductance],[outsidegapconductance],[hardness]], 'xaxis':[xaxis]}

def helper_int(w):
	XX = w['XX']#+1e3/20
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])

	#if XX.ndim == 1: #if 1D 
	#	XX = np.cumsum(XX)
	#	XX = np.cumsum(XX,XX[-1])
	for i in range(0,xn):
		intarr = np.cumsum(XX[i,:])
		XX[i,:] = intarr - intarr[yn/2] #correct for integration constant
	#1 nA / 1 mV = 0.0129064037 conductance quanta
	w['XX'] = XX * w['ystep']# * 0.0129064037
	#w['cbar_quantity'] = 'intI'
	#w['cbar_unit'] = '?'
	#w['cbar_unit'] = r'2$\mathrm{e}^2/\mathrm{h}$'	

def helper_sgdidv(w):
	'''Perform Savitzky-Golay smoothing and get 1st derivative'''
	ylabel = w['ylabel']
	xlabel = w['xlabel']
	cbar_q = w['cbar_quantity']
	cbar_u = w['cbar_unit']
	w['XX'] = signal.savgol_filter(w['XX'], int(w['sgdidv_samples']), int(w['sgdidv_order']), deriv=1, delta=w['ystep'])# / 0.02581281)
	
	w['cbar_quantity'] = '$\partial$' + cbar_q + '/$\partial$' + ylabel.split(' ', 1)[0]
	
	if cbar_u == 'nA':
		#1 nA / 1 mV = 0.0129064037 conductance quanta
		w['XX'] = w['XX'] / w['ystep']# * 0.0129064037
		#w['cbar_unit'] = '$\mu$Siemens'
		w['cbar_unit'] = r'$\mathrm{e}^2/\mathrm{h}$'
	elif cbar_u == 'mV':
		#1 mV / 1 nA = 77.4809173 conductance resistance
		w['XX'] = w['XX'] / w['ystep']# * 77.4809173
		#w['cbar_unit'] = '$M\Omega'
		w['cbar_unit'] = r'$\mathrm{h}/\mathrm{e}^2$'
	else:
		w['XX'] = w['XX'] / w['ystep']# * 0.0129064037
		w['cbar_unit'] = ''

def helper_sgtwodidv(w):
	'''Perform Savitzky-Golay smoothing and get 1st derivative'''
	w['XX'] = signal.savgol_filter(
			w['XX'], int(w['sgdidv_samples']), int(w['sgdidv_order']), deriv=1, delta=w['ystep'] / 0.0129064037)
	w['cbar_quantity'] = 'dI/dV'
	w['cbar_unit'] = '$\mu$Siemens'
	w['cbar_unit'] = r'$\mathrm{e}^2/\mathrm{h}$'
	data = w['XX']

def helper_log(w):
	w['XX'] = np.log10(np.abs(w['XX']))
	w['cbar_trans'] = ['log$_{10}$','abs'] + w['cbar_trans']
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
	w['XX'] = w['XX']
	
def helper_movingmeansubtract(w):
	XX = w['XX']
	xn, yn = XX.shape
	meanarray = np.zeros(xn)
	for i in range(0,xn):
		meanarray[i] = np.mean(XX[i][:])
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
	
def helper_meansubtract(w):
	offset = np.mean(w['XX'])
	print('Subtracted mean:' + str(offset))
	w['XX'] = w['XX']-offset

def helper_deleteouterdatapoints(w):
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

def helper_offsetslopesubtract(w):
	offset,slope = (w['offsetslopesubtract_offset']),(w['offsetslopesubtract_slope'])
	#print offset
	xaxis = np.linspace(w['ext'][1],w['ext'][2],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	ymatrix = np.repeat([yaxis],w['XX'].shape[0], axis = 0)
	w['XX'] = w['XX']-(ymatrix*slope)-offset

def helper_vbiascorrector(w):
	import math
	from matplotlib.mlab import griddata
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
	ylimitneg,ylimitpos = math.floor(np.amin(ycorrected*10))/10, math.ceil(np.amax(ycorrected*10))/10
	gridyaxis = np.linspace(ylimitneg,ylimitpos,int(yn*gridresolutionfactor))
	gridxaxis = xaxis
	XX_new = np.zeros(shape=(xn,len(gridyaxis)))
	if w['vbiascorrector_didv'] == True: # Calls didv within helper_vbiascorrector before interpolation to prevent artefacts.
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

def helper_ivreverser(w): #Inverse I and V-bias measurements (works on both) by interpolating y-data on new homogeneous x-axis.
	import math
	from matplotlib.mlab import griddata
	import numpy.ma as ma
	import numpy as np
	XX = w['XX']#+1e3/float(w['vbiascorrector_seriesr'])
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	ycorrected = np.zeros(shape=(xn,yn))
	gridresolutionfactor = int(w['ivreverser_gridresolutionfactor']) # Example: Factor of 2 doubles the number of y datapoints for non-linear interpolation
	
	for i in range(0,xn):
		ycorrected[i,:] = XX[i,:] #y-axis becomes data axis
		XX[i,:] = yaxis #data axis becomes y-axis
		#datacorrected[i,:] = yaxis #data axis becomes y-axis
	ylimitneg,ylimitpos = math.floor(np.amin(ycorrected*10))/10, math.ceil(np.amax(ycorrected*10))/10
	gridyaxis = np.linspace(ylimitneg,ylimitpos,int(yn*gridresolutionfactor))
	gridxaxis = xaxis
	XX_new = np.zeros(shape=(xn,len(gridyaxis)))
	for i in range(0,xn):
		XX_new[i,:] = np.interp(gridyaxis,ycorrected[i,:],XX[i,:], left=np.nan, right=np.nan)
	w['XX'] = XX_new
	#w['ext'][2] = ylimitneg
	#w['ext'][3] = ylimitpos
	w['ext'] = (w['ext'][0],w['ext'][1],ylimitneg,ylimitpos)
	w['ystep'] = (ylimitpos-ylimitneg)/(len(gridyaxis)-1)
	#print ylimitpos, ylimitneg, len(gridyaxis)
	print('new ystep:'+ str(w['ystep']))
	w['ext'] = (w['ext'][0],w['ext'][1],ylimitneg,ylimitpos)
	if w['ylabel'].find('nA') != -1:
		print('I sourced detected')
		w['ylabel'] = '$V_\mathrm{SD}$ (mV)'
		w['cbar_quantity'] = '$I_\mathrm{S}$'
		w['cbar_unit'] = 'nA'
	elif w['ylabel'].find('mV') != -1:
		print('V sourced detected')
		w['ylabel'] = '$I_\mathrm{D}$ (nA)'
		w['cbar_quantity'] = '$V_\mathrm{SD}$'
		w['cbar_unit'] = 'mV' 

def helper_excesscurrent(w): #Designed for I-bias. Calculate excess current by performing linear fit at high bias and calculate the zero-crossing of the x-axis
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
	xlabel = w['xlabel']
	ylabel = w['ylabel']
	plt.plot(xaxis,excesscurrpos,xaxis,excesscurrneg)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	w['buffer']={'labels': [xlabel,ylabel], 'data':[dataarray], 'xaxis':[xaxis], 'measdata':[XX]} #wrapping for analaysis
	
def helper_linecut(w): #Make a linecut on specified axis and value.
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	linecutvalue = w['linecut_value']
	#print type(linecutvalue)
	if type(w['linecut_value']) is str: #If multiple linecuts are given separated by '_', unpack.
		linecutvalue = w['linecut_value'].split('_')
	else: 
		linecutvalue = [w['linecut_value']]
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
		xlabel = w['ylabel']
		ylabel = w['cbar_quantity'] + ' ('+ w['cbar_unit'] + ')'
		xaxis = yaxis
	if axis == 'y':
		dataarray = np.zeros((len(linecutvalue),xn))
		for val in range(len(linecutvalue)):
			yindex = np.abs(yaxis - (linecutvalue[val])).argmin()
			datavals = XX[:,yindex]
			plt.plot(xaxis,datavals)
			dataarray[val,:] = datavals
		xlabel = w['xlabel']
		ylabel = w['cbar_quantity'] + ' ('+ w['cbar_unit'] + ')'
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	w['buffer']={'labels': [xlabel,ylabel], 'data':[dataarray], 'xaxis':[xaxis], 'vals':[linecutvalue]} #wrapping for further analysis


def helper_resistance(w): #Calculate resistance of one sweep. Pretty old, not sure if it still works.
	XX = w['XX']
	xn, yn = XX.shape
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	linecutvalue = w['resistance_linecutvalue']
	dopolyfit = w['resistance_dolinearfit']
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

def helper_dbmtovolt(w): #Convert dmb to 'volt', rf-amplifier (type?) conversion table included. Interpolates linear to logarithmic power axis.
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

	if w['dbmtovolt_rfamp']==True: # Creates new power axis using attenuation and optional rf-amp
		xaxislin = dbmtovolt(linrfamp(xaxis), w['dbmtovolt_attenuation'])
	else:
		xaxislin = dbmtovolt(xaxis, w['dbmtovolt_attenuation'])
		ycorrected = np.zeros(shape=(xn,yn))
	
	gridresolutionfactor = int(w['dbmtovolt_gridresolutionfactor']) #Factor multiplying original resolution for interpolation of new power axis.
	xlimitneg,xlimitpos = np.amin(xaxislin), np.amax(xaxislin)
	gridxaxis = np.linspace(xlimitneg,xlimitpos,int(yn*gridresolutionfactor))
	gridyaxis = yaxis
	XX_new = np.zeros(shape=(len(gridxaxis),len(gridyaxis)))
	for i in range(0,yn):
		XX_new[:,i] = np.interp(gridxaxis,xaxislin,XX[:,i], left=np.nan, right=np.nan)
	w['XX'] = XX_new
	#w['ext'][2] = ylimitneg
	#w['ext'][3] = ylimitpos
	w['ext'] = (xlimitneg,xlimitpos,w['ext'][2],w['ext'][3])
	w['xstep'] = (xlimitpos-xlimitneg)/len(gridxaxis)
	#print ylimitpos, ylimitneg, len(gridyaxis)
	print('new xstep:'+ str(w['xstep']))
	w['xlabel'] = '$V_\mathrm{rms}$ (V)'


def helper_shapiro(w): #Looks for expected voltage of Shapiro step as function of applied frequency and order and returns closest measured index.
	planck = 4.135668e-15
	electron = 1#1.60e-19
	rffreq = w['shapiro_rffreq'] #Freq in GHz
	nsteps = int(w['shapiro_nsteps']) #Up to which order
	w['XX'] = w['XX']*2*electron/(planck*rffreq*1000) #Convert XX to energy
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
	xlabel = w['xlabel']
	#ylabel = w['cbar_quantity'] + ' ('+ w['cbar_unit'] + ')'
	ylabel = w['ylabel']
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	#print steppos.shape
	shapiro = [steppos,stepneg]
	w['buffer'] = shapiro
	w['buffer']={'labels': [xlabel,ylabel], 'data':[shapiro], 'xaxis':[xaxis]}

def helper_histogram(w): #Make a histogram
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

def helper_histogram2(w): #Make histogram using stats.binned_statistics. Not sure what the advantage was.
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
		# for j in range(0,histbins):
			# inds = np.where(stats[2][:] == j+1)[0]
			# xaxvals = [xaxis[k] for k in inds]
			# if xaxvals:
				# step[j] = argmax(xaxvals) - argmin(xaxvals)
			# else:
				# step[j] = 0

def helper_minsubtract(w): #Subtract smallest value from data
	w['XX'] = w['XX'] - np.min(w['XX'])

def helper_factor(w): #Multiply data with a factor
	w['XX'] = w['XX']*float(w['factor_factor'])
	#w['cbar_trans'] = ['abs'] + w['cbar_trans']
	#w['cbar_quantity'] = '|' +w['cbar_quantity']+'|'
	#w['cbar_unit'] = w['cbar_unit']

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

def helper_deint_cross(w):
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

def helper_ic(w): #For meander + SC measurements: split file at zero bias and use 0 to finite sweep direction values for
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
    #print top.shape
    #print bottom.shape
    w['XX']=np.hstack((top,bottom))
    #dinges = np.hstack((top,bottom))
    #w['XX'] = bottom
    # print bottom.shape
    # print top.shape

def helper_iretrap(w): #For meander + SC measurements: split file at zero bias and use finite to 0 sweep direction values (different 
	# from helper_ic) values for both pos and neg bias. Next stick them together again.
    data = w['XX']
    helper_deinterlace(w)
    X0 = w['deinterXXodd']
    X1 = w['deinterXXeven']
    xn,yn = data.shape
    #pak de onderkant van X0
    bottom = X0[:,yn/2:]
    #pak de bovenkant van X1

    # even/odd statements work on even deinterlacing (X1) since this is the largest
    if (yn % 2 == 0):
        cutt = yn/2-1
    else:
        cutt = yn/2
    #take the op of X0 (even) and check for even/uneven
    if(xn % 2 == 0):
        top = X1[:,:cutt]
    else:
        top = X1[1:,:cutt]
    #top = X1[1:,:yn/2]
    w['XX']=np.hstack((top,bottom))

def helper_icvsx(w): #Finds switching current or retrapping current by peak fitting. Preparation of data required by various other styles.
	# Correct use is complicated and requires lots of tuning (see comments). Styles to use before: 
	import sys
	from IPython.display import display
	import peakutils
	useonlythreshold = w['icvsx_useonlythreshold']
	pixelnoiserange = int(w['icvsx_pixelnoiserange'])
	peaktoplateauthreshold = w['icvsx_ppt'] # threshold
	stepoffset = w['icvsx_stepoffset']
	strictzero = w['icvsx_strictzero']
	limhigh = float(w['icvsx_plateaulim'])
	limmin = float(w['icvsx_gapmax'])
	print(useonlythreshold,pixelnoiserange,peaktoplateauthreshold,stepoffset,strictzero)
	# def reject_outliers(data, m = 2.):
		# d = np.abs(data - np.median(data))
		# mdev = np.median(d)
		# s = d/mdev if mdev else 0.
		# return data[s<m]
		
	#from peakutils.plot import plot as pplot
	from matplotlib import pyplot
	
	XX = w['XX']
	xn, yn = XX.shape
	print(yn,xn)
	xaxis = np.linspace(w['ext'][0],w['ext'][1],w['XX'].shape[0])
	yaxis = np.linspace(w['ext'][2],w['ext'][3],w['XX'].shape[1])
	ic = np.zeros((3,xn))
	indexes = np.zeros(2)
	
	for i in range(0,xn):
		#XX[i,:] = XX[i,:]-peakutils.baseline(XX[i,:],2)/2
		posarray = XX[i,int((yn/2)-1+stepoffset):0:-1]
		#posarrayhighbaseline = peakutils.baseline(posarray,1)/2
		#posarray = posarray-posarrayhighbaseline
		
		negarray = XX[i,int((yn/2)+stepoffset):yn]
		#negarrayhighbaseline = peakutils.baseline(negarray,1)/2
		#print negarray
		#print negarrayhighbaseline 
		#negarray = negarray-negarrayhighbaseline
		
		fullarray = XX[i,:]
		#print negarray
		posmax,negmax = np.amax(posarray),np.amax(negarray)
		posmin,negmin = np.amin(posarray),np.amin(negarray)
		poshigh,neghigh = np.mean(posarray[-20:-5]),np.mean(negarray[-20:-5])
		if poshigh>limhigh:
			poshigh=limhigh
		if neghigh>limhigh:
			neghigh=limhigh
		if strictzero == True:
			poslow,neglow = 0,0
		else:
			poslow,neglow = np.mean((posarray[0:3]+negarray[0:3])/2),np.mean((posarray[0:3]+negarray[0:3])/2)
			if poslow>limmin:
				print('minlimfix')
				poslow=limmin
			if neglow>limmin:
				neglow=limmin
		#print neghigh,poshigh
		#print neglow,poslow
		#print posarray[0:3]+negarray[0:3]
		#print poslow,neglow
		thresposabs = (poshigh - poslow)*peaktoplateauthreshold + poslow
		thresnegabs = (neghigh - neglow)*peaktoplateauthreshold + neglow
		#print 'poshighlow: ' + str(poshigh) + ' \ '  + str(poslow)
		#print 'neghighlow: ' + str(neghigh) + ' \ '  + str(neglow)
		#print 'thresholdposneg: ' + str(thresposabs) + ' \ '  +str(thresnegabs)
		threspos = (thresposabs-posmin)/(posmax-posmin)
		thresneg = (thresnegabs-negmin)/(negmax-negmin)
		#print 'posmaxmin: ' + str(posmax) + ' \ '  + str(posmin)
		#print 'negmaxmin: ' + str(negmax) + ' \ '  + str(negmin)
		#print 'normalised thres: ' + str(threspos) + ' \ ' + str(thresneg)
		#print posmax-posmin
		#print thresposabs-posmin
		indexneg = peakutils.indexes(negarray,thres=thresneg, min_dist=1)
		# if indexneg.size == 0:
			# indexneg = np.array([0])
		indexpos = peakutils.indexes(posarray,thres=threspos, min_dist=1)
		# if indexpos.size == 0:
			# indexpos = np.array([0])
		try:
			indexnegnopeak = [x[0] for x in enumerate(negarray) if x[1] > thresnegabs]
			if len(indexnegnopeak) == 0:
				indexnegnopeak = np.array([0])
		except:
			e = sys.exc_info()[0]
			print(e)
			print('exceptneg')#indexnegnopeak = np.array([0]))
		try:
			indexposnopeak = [x[0] for x in enumerate(posarray) if x[1] > thresposabs]
			if len(indexposnopeak) == 0:
				indexposnopeak = np.array([0])
		except:
			e = sys.exc_info()[0]
			print(e)
			print('exceptpos')#indexposnopeak = np.array([0])
			
		if useonlythreshold == 1:
			indexneg = indexnegnopeak
			indexpos = indexposnopeak

		firstpeakneg = indexneg[0]
		peaknoise = True
		counter = 0
		while peaknoise == True and counter < 1000:
			counter = counter + 1
			try:
				if True in (negarray[firstpeakneg:firstpeakneg+pixelnoiserange] < (-np.mean(negarray)/5)):
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
				#if True in (posarray[firstpeakpos:firstpeakpos+pixelnoiserange] < (-np.mean(posarray)/5)):
				if True in (posarray[firstpeakpos:firstpeakpos+pixelnoiserange] < ( -posarray[firstpeakpos]/10)):
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
		#print finalindexpos,finalindexneg
		#if sindex[0] > 2:
		#print 'Index '+ str(i) + ' has more than two peaks' + str(indexes)
		#print yaxis
		if finalindexneg == 0:
			ic[0,i] = 0
		else:
			ic[0,i] = (yaxis[finalindexneg+yn/2]) #neg one
		if finalindexpos == 0:
			ic[1,i] = 0
		else:
			ic[1,i] = (yaxis[finalindexpos+yn/2]) #pos one
		#print ic[1,i]
	#print ic[2,:]
	#w['buffer']={'labels': [xlabel,ylabel], 'data':[dataarray], 'xaxis':[xaxis]}
	fig = plt.figure()
	xlabel = w['xlabel']
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
	'deinterlace': helper_deinterlace,
	'deinterlace0': helper_deinterlace0,
	'deinterlace1': helper_deinterlace1,
	'didv': helper_didv,
	'log': helper_log,
	'normal': helper_normal,
	#'ssidrive': helper_ssidrive,
	'flipaxes': helper_flipaxes,
	'flipxaxis': helper_flipxaxis,
	'flipyaxis': helper_flipyaxis,
	'mov_avg': helper_mov_avg,
	'abs': helper_abs,
	'savgol': helper_savgol,
	'sgdidv': helper_sgdidv,
	'sgtwodidv': helper_sgtwodidv,
	'fancylog': helper_fancylog,
	'minsubtract': helper_minsubtract,
	'crosscorr':helper_crosscorr,
# 	'threshold_offset': helper_threshold_offset,
	'massage': helper_massage,
	'deint_cross': helper_deint_cross,
	'shapiro': helper_shapiro,
	'meansubtract': helper_meansubtract,
	'movingmeansubtract': helper_movingmeansubtract,
	'offsetslopesubtract': helper_offsetslopesubtract,
	'ic': helper_ic,
	'iretrap': helper_iretrap,
	'icvsx': helper_icvsx,
	'excesscurrent': helper_excesscurrent,
	'resistance': helper_resistance,
	'vbiascorrector': helper_vbiascorrector,
	'ivreverser': helper_ivreverser,
	'histogram': helper_histogram,
	'int': helper_int,
	'fixlabels': helper_fixlabels,
	'hardgap': helper_hardgap,
	'changeaxis': helper_changeaxis,
	'linecut': helper_linecut,
	'dbmtovolt': helper_dbmtovolt,
	'deleteouterdatapoints': helper_deleteouterdatapoints,
	'factor': helper_factor
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
	'deinterlace': {'param_order': []},
	'deinterlace0': {'param_order': []},
	'deinterlace1': {'param_order': []},
	'didv': {'condquant': False, 'param_order': ['condquant']},
	'log': {'param_order': []},
	'normal': {'param_order': []},
	'ssidrive': {'param_order': []},
	'flipaxes': {'param_order': []},
	'flipyaxis': {'param_order': []},
	'flipxaxis': {'param_order': []},
# 	'threshold_offset': {'threshold':0.2,'start':0.0,'stop':1.0,'param_order':[]},
	'mov_avg': {'m': 1, 'n': 3, 'win': None, 'param_order': ['m', 'n', 'win']},
	'abs': {'param_order': []},
	'savgol': {'samples': 11, 'order': 3, 'param_order': ['samples', 'order']},
	'sgdidv': {'samples': 11, 'order': 3, 'param_order': ['samples', 'order']},
	'sgtwodidv': {'samples': 21, 'order': 3, 'param_order': ['samples', 'order']},
	'fancylog': {'cmin': None, 'cmax': None, 'param_order': ['cmin', 'cmax']},
	'minsubtract': {'param_order': []},
	'crosscorr': {'peakmin':None,'peakmax':None,'toFirstColumn':True,'param_order': ['peakmin','peakmax','toFirstColumn']},
	'massage': {'param_order': []},
	'deint_cross': {'param_order': []},
	'shapiro': {'rffreq': 2.15e9, 'nsteps': 1, 'param_order': ['rffreq','nsteps']},
	'meansubtract': {'param_order': []},
	'movingmeansubtract': {'window': 2,'param_order': ['window']},
	'offsetslopesubtract': {'slope': 0, 'offset': 0, 'param_order': ['slope', 'offset']},
	'ic': {'param_order': []},
	'iretrap': {'param_order': []},
	'icvsx': {'useonlythreshold': True, 'pixelnoiserange': 3, 'ppt': 0.5, 'stepoffset': 0, 'strictzero': True, 'plateaulim': 1e6,'gapmax': 1e6,'param_order': ['useonlythreshold','pixelnoiserange','ppt','stepoffset','strictzero','plateaulim','gapmax']},
	'excesscurrent': {'datacutoff': 3, 'rangefactor': 0.15, 'plot': 0, 'plotval': 0,'param_order': ['datacutoff','rangefactor','plot','plotval']},
	'resistance': {'linecutvalue': 0, 'dolinearfit': False, 'fitregion': 1, 'param_order': ['linecutvalue','dolinearfit','fitregion']},
	'vbiascorrector':{'voffset': 0,'seriesr': 0, 'gridresolutionfactor': 1, 'didv':False, 'param_order': ['voffset','seriesr','gridresolutionfactor','didv']},
	'ivreverser':{'gridresolutionfactor': 1, 'param_order': ['gridresolutionfactor']},
	'histogram':{'bins': 25, 'rangemin': -1, 'rangemax': 1, 'param_order': ['bins','rangemin','rangemax']},
	'int': {'param_order': []},
	'fixlabels': {'param_order': []},
	'hardgap': {'gaprange': 0.1, 'outsidegapmin': 0.5, 'outsidegapmax': 0.6, 'param_order': ['gaprange','outsidegapmin','outsidegapmax']},
	'changeaxis': {'xfactor': 1, 'yfactor': 1, 'datafactor': 1, 'dataunit': None, 'xunit': None, 'yunit': None, 'param_order': ['xfactor','yfactor','xunit','yunit', 'datafactor', 'dataunit']},
	'linecut': {'linecutvalue': 1,'axis': None,'param_order': ['linecutvalue','axis']},
	'dbmtovolt': {'rfamp': False, 'attenuation': 0, 'gridresolutionfactor': 2, 'param_order': ['rfamp','attenuation','gridresolutionfactor']},
	'deleteouterdatapoints': {'n':0,'param_order': ['n']},
	'factor': {'factor':1,'param_order': ['factor']}

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
		'ystep':1,
		'X': [],
		'XX': [], 
		'cbar_quantity': '',
		'cbar_unit': 'a.u.',
		'cbar_trans': [],
		'xlabel': [],
		'ylabel': [],
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
    return signal.convolve2d(data, window, mode='same', boundary='symm')

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
    return signal.convolve(data, window, mode='same')


def get_offset(x,y1,y2):
#     plt.figure(43)
#     plt.plot(x,y1,x,y2)
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
    
    
#     plt.figure(44)
#     plt.plot(corr,'o')
#     plt.plot(xcorrfit,result.best_fit,'-')
#     plt.plot(xcorrfit,result.init_fit,'-')

    x0=result.best_values['x0']
    span = max(x)-min(x)
    #map back to real x values
    xmap= np.linspace(
				span/2, -span/2
				,len(xcorr)) 
    offset_intp = np.interp(x0,xcorr,xmap)
    return offset_intp