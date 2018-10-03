import numpy as np
from scipy import signal
import re
import collections
import matplotlib.colors as mplc

REGEX_STYLE_WITH_PARAMS = re.compile('(.+)\((.+)\)')
REGEX_VARVALPAIR = re.compile('(\w+)=(.*)')

def nonzeromin(x):
	'''
	Get the smallest non-zero value from an array
	Also works recursively for multi-dimensional arrays
	Returns None if no non-zero values are found
	'''
	x = numpy.array(x)
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
	

def helper_savgol(w):
	'''Perform Savitzky-Golay smoothing'''
	w['XX'] = signal.savgol_filter(
			w['XX'], int(w['savgol_samples']), int(w['savgol_order']))

def helper_didv(w):
	a=(w['XX'])

	if a.ndim == 1: #if 1D 
		w['XX'] = np.diff(w['XX'])
		w['XX'] = np.append(w['XX'],w['XX'][-1])
	else:
		w['XX'] = np.diff(w['XX'],axis=1)
	#1 nA / 1 mV = 0.02581281 * (e^2/h) #conversion for siemens to half a conductance quantum e^2/h
	w['XX'] = w['XX'] / w['ystep'] * 0.02581281
	w['cbar_quantity'] = 'dI/dV'
	w['cbar_unit'] = '$\mu$Siemens'
	w['cbar_unit'] = r'$\mathrm{e}^2/\mathrm{h}$'

def helper_sgdidv(w):
	'''Perform Savitzky-Golay smoothing and get 1st derivative'''
	w['XX'] = signal.savgol_filter(
			w['XX'], int(w['sgdidv_samples']), int(w['sgdidv_order']), deriv=1, delta=w['ystep'] / 0.02581281)
	w['cbar_quantity'] = 'dI/dV'
	w['cbar_unit'] = '$\mu$Siemens'
	w['cbar_unit'] = r'$\mathrm{e}^2/\mathrm{h}$'


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
	This might be incompatible with Fiddle.
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
	
def helper_minsubtract(w):
	w['XX'] = w['XX'] - np.min(w['XX'])


def helper_abs(w):
	w['XX'] = np.abs(w['XX'])
	w['cbar_trans'] = ['abs'] + w['cbar_trans']
	w['cbar_quantity'] = w['cbar_quantity']
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



def helper_massage(w):
	func = w['massage_func']
	func(w)

STYLE_FUNCS = {
	'deinterlace': helper_deinterlace,
	'deinterlace0': helper_deinterlace0,
	'deinterlace1': helper_deinterlace1,
	'didv': helper_didv,
	'log': helper_log,
	'normal': helper_normal,
	'flipaxes': helper_flipaxes,
	'flipxaxis': helper_flipxaxis,
	'flipyaxis': helper_flipyaxis,
	'mov_avg': helper_mov_avg,
	'abs': helper_abs,
	'savgol': helper_savgol,
	'sgdidv': helper_sgdidv,
	'fancylog': helper_fancylog,
	'minsubtract': helper_minsubtract,
	'crosscorr':helper_crosscorr,
# 	'threshold_offset': helper_threshold_offset,
	'massage': helper_massage,
	'deint_cross': helper_deint_cross
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
	'didv': {'param_order': []},
	'log': {'param_order': []},
	'normal': {'param_order': []},
	'flipaxes': {'param_order': []},
	'flipyaxis': {'param_order': []},
	'flipxaxis': {'param_order': []},
# 	'threshold_offset': {'threshold':0.2,'start':0.0,'stop':1.0,'param_order':[]},
	'mov_avg': {'m': 1, 'n': 5, 'win': None, 'param_order': ['m', 'n', 'win']},
	'abs': {'param_order': []},
	'savgol': {'samples': 11, 'order': 3, 'param_order': ['samples', 'order']},
	'sgdidv': {'samples': 11, 'order': 3, 'param_order': ['samples', 'order']},
	'fancylog': {'cmin': None, 'cmax': None, 'param_order': ['cmin', 'cmax']},
	'minsubtract': {'param_order': []},
	'crosscorr': {'peakmin':None,'peakmax':None,'toFirstColumn':True,'param_order': ['peakmin','peakmax','toFirstColumn']},
	'massage': {'param_order': []},
	'deint_cross': {'param_order': []}
}

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
						if val.lower() == 'true':
							val = True
						elif val.lower() == 'false':
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
			#print(s)
			func = STYLE_FUNCS[s.split('(')[0]]
			func(wrap)
			#print(wrap)
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
	# Makes sure data array is a numpy array or masked array.
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
	# Makes sure data array is a numpy array or masked array.
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