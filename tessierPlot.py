#tessier.py
#tools for plotting all kinds of files, with fiddle control etc

# data = loadFile(...)
# a = plot3DSlices(data,)

import matplotlib.pyplot as plt
import matplotlib as mpl


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.widgets import Button,Line2D
from scipy.signal import argrelmax
import matplotlib.gridspec as gridspec


import pandas as pd
import numpy as np
import math
import tessierView as tv

import re
import os
_moduledir = os.path.dirname(os.path.realpath(__file__))

from tessierGuiElements import *
import tessierStyles as tstyle
from imp import reload #DEBUG
reload(tstyle) #DEBUG


##Only load this part on first import, calling this on reload has dire consequences
## Note: there is still a bug where closing a previously plotted window and plotting another plot causes the window and the kernel to hang
try:
	magichappened
except:
	import IPython  
	ipy=IPython.get_ipython()
	ipy.magic("pylab qt")
	magichappened = False
else:
	magichappened = True
from mpl_toolkits.axes_grid1 import make_axes_locatable



def parseUnitAndNameFromColumnName(input):
	reg = re.compile(r'\{(.*?)\}')
	z = reg.findall(input)
	return z

def getnDimFromData(file,data):

		nDim = 0
		names,skiprows,header=parseheader(file)

		cols = data.columns.tolist()
		filterdata = data.sort(cols[:-1])
		filterdata = filterdata.dropna(how='any')
		#first determine the columns belong to the axes (not measure) coordinates
		cols = [i for i in header if (i['type'] == 'coordinate')]

		for i in cols:
			col = getattr(filterdata,i['name'])
			if len(col.unique()) > 1: #do not count 0d axes..(i.e. with only 1 unique value)
				nDim += 1
# 			self.uniques_per_col.append(list(col.unique()))
		return nDim
	


def loadCustomColormap(file='./cube1.txt'):
	do = np.loadtxt(file)
	ccmap = mpl.colors.LinearSegmentedColormap.from_list('name',do)

	ccmap.set_under(do[0])
	ccmap.set_over(do[-1])
	return ccmap



def buildLogicals(xs):
#combine the logical uniques of each column into boolean index over those columns
#infers that each column has
#like
# 1, 3
# 1, 4
# 1, 5
# 2, 3
# 2, 4
# 2, 5
#uniques of first column [1,2], second column [3,4,5]
#go through list and recursively combine all unique values
	if len(xs) > 1:
		for i in xs[0].unique():
			if math.isnan(i):
				continue
			for j in buildLogicals(xs[1:]):
				yield (xs[0] == i) & j ## boolean and
	elif len(xs) == 1:
		for i in xs[0].unique():
			if (math.isnan(i)):
				continue
			yield xs[0] == i
	else:
		#empty list
		yield slice(None) #return a 'semicolon' to select all the values when there's no value to filter on


class plotR:
	def __init__(self,file):
		self.header = None
		self.names = None
		self.fig = None
	
		self.exportData =[]	
		self.exportDataMeta = []	
		self.file = file

		self.data  = self.loadFile(file) 
		self.name  = file
		self.filterdata = None
		
		self.ndim = self.getnDimFromData()
		self.dims = self.getdimFromData()
		self.bControls = True #boolean controlling state of plot manipulation buttons
		

	def is2d(self,**kwargs):
		nDim = self.ndim
		#if the uniques of a dimension is less than x, plot in sequential 2d, otherwise 3d

		#maybe put logic here to plot some uniques as well from nonsequential axes?
		filter = self.dims < 5
		filter_neg = np.array([not x for x in filter])

		coords = np.array([x['name'] for x in self.header if x['type']=='coordinate'])
		
		return len(coords[filter_neg]) < 2
	
	def quickplot(self,**kwargs):
		nDim = self.ndim

		filter = self.dims < 5

		coords = np.array([x['name'] for x in self.header if x['type']=='coordinate'])
		
		uniques_col_str = coords[filter]
		print uniques_col_str

		if self.is2d(): 
			fig = self.plot2d(**kwargs)
		else:
			fig = self.plot3d(uniques_col_str=uniques_col_str,**kwargs)
			self.exportToMtx() #do thi
		
		return fig

	def autoColorScale(self,data):
		values, edges = np.histogram(data, 256)
		maxima = edges[argrelmax(values,order=24)]
		if maxima.size>0: 
			cminlim , cmaxlim = maxima[0] , np.max(data)
		else:
			cminlim , cmaxlim = np.min(data) , np.max(data)
		return (cminlim,cmaxlim)

		
	def plot3d(self,	massage_func=None,
						uniques_col_str=[],
						drawCbar=True,
						subplots_args={'top':0.96, 'bottom':0.17, 'left':0.14, 'right':0.85,'hspace':0.0},
						ax_destination=None,
						n_index=None,
						style='log',
						clim=None,
						aspect='auto',
						interpolation='none',
						value_axis=-1,
						sweepoverride=False,
						cbar_orientation='vertical',
						cbar_location ='normal',
						**kwargs):
		if not self.fig:
			self.fig = plt.figure()
		
		if self.data is None:
			self.loadFile(file)
			
		cols = self.data.columns.tolist()
		filterdata = self.sortdata()
		
		uniques_col = []
		self.uniques_per_col=[]
		self.data = self.data.dropna(how='any')

				

		for i in uniques_col_str:
			col = getattr(filterdata,i)
			uniques_col.append(col)
			self.uniques_per_col.append(list(col.unique()))
		
		self.ccmap = loadCustomColormap()


		self.fig.subplots_adjust(**subplots_args)


		n_subplots = 1
		for i in self.uniques_per_col:
			n_subplots *= len(i)

		if n_index is not None:
			n_index = np.array(n_index)
			n_subplots = len(n_index)
			
		if n_subplots > 1:
			width = 2
		else:
			width = 1
		n_valueaxes = len(self.get_valuekeys())
		if n_valueaxes >1:
			width = n_valueaxes
			
		gs = gridspec.GridSpec(int(n_subplots/width)+n_subplots%width, width)
		
		cnt=0
		#enumerate over the generated list of unique values specified in the uniques columns
		for j,ind in enumerate(buildLogicals(uniques_col)):
			#
			for value_axis in range(n_valueaxes):
				
				#plot only if number of the plot is indicated
				if n_index is not None:
					if j not in n_index:
						continue
				data_byuniques = filterdata.loc[ind]
				data_slice = data_byuniques

				#get the columns /not/ corresponding to uniques_cols
				#find the coord_keys in the header
				coord_keys = self.get_coordkeys()
			
				#filter out the keys corresponding to unique value columns
				us=uniques_col_str		
				coord_keys = [key for key in coord_keys if key not in uniques_col_str ]
				#now find out if there are multiple value axes
				value_keys = self.get_valuekeys()
			
				x=data_slice.loc[:,coord_keys[-2]]
				y=data_slice.loc[:,coord_keys[-1]]
				z=data_slice.loc[:,value_keys[value_axis]]
			
				xu = np.size(x.unique())
				yu = np.size(y.unique())
			

				## if the measurement is not complete this will probably fail so trim of the final sweep?
				print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))

				if xu*yu != len(z):
					xu = (len(z) / yu) #dividing integers so should automatically floor the value

				#trim the first part of the sweep, for different min max, better to trim last part?
				#or the first since there has been sorting
				#this doesnt work for e.g. a hilbert measurement

				print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,len(z)))

				#sorting sorts negative to positive, so beware:
				#sweep direction determines which part of array should be cut off
				if sweepoverride: ##if sweep True, override the detect value
					z = z[-xu*yu:]
					x = x[-xu*yu:]
					y = y[-xu*yu:]
				else:
					z = z[:xu*yu]
					x = x[:xu*yu]
					y = y[:xu*yu]

				XX = np.reshape(z,(xu,yu))

				self.x = x
				self.y = y
				self.z = z
				#now set the lims
				xlims = (x.min(),x.max())
				ylims = (y.min(),y.max())

				#determine stepsize for di/dv, inprincipe only y step is used (ie. the diff is also taken in this direction and the measurement swept..)
				xstep = (xlims[0] - xlims[1])/xu
				ystep = (ylims[0] - ylims[1])/yu
				ext = xlims+ylims
				self.extent = ext
				self.XX = XX
			
				self.exportData.append(XX)
				try:
					m={
						'xu':xu,
						'yu':yu,
						'xlims':xlims,
						'ylims':ylims,
						'zlims':(0,0),
						'xname':cols[-3],
						'yname':cols[-2],
						'zname':'unused',
						'datasetname':self.name}
					self.exportDataMeta = np.append(self.exportDataMeta,m)
				except Exception,e:
					print e
					pass
				if ax_destination is None:
					ax = plt.subplot(gs[cnt])
				else:
					ax = ax_destination
				cbar_title = ''
			

			
				if type(style) != list:
					style = list([style])


				measAxisDesignation = parseUnitAndNameFromColumnName(value_keys[value_axis])
				#wrap all needed arguments in a datastructure
				cbar_quantity = measAxisDesignation[0]
				cbar_unit = measAxisDesignation[1]
				cbar_trans = [] #trascendental tracer :P For keeping track of logs and stuff

				w = tstyle.getPopulatedWrap(style)
				w2 = {'ext':ext, 'ystep':ystep,'XX': XX, 'cbar_quantity': cbar_quantity, 'cbar_unit': cbar_unit, 'cbar_trans':cbar_trans}
				for k in w2:
					w[k] = w2[k]
				w['massage_func']=massage_func
				tstyle.processStyle(style, w)

				#unwrap
				ext = w['ext']
				XX = w['XX']
				cbar_trans_formatted = ''.join([''.join(s+'(') for s in w['cbar_trans']])
				cbar_title = cbar_trans_formatted + w['cbar_quantity'] + ' (' + w['cbar_unit'] + ')'
				if len(w['cbar_trans']) is not 0:
					cbar_title = cbar_title + ')'
			
			
				#postrotate np.rot90
				XX = np.rot90(XX)
			
				if 'deinterlace' in style:
					self.fig = plt.figure()
					ax_deinter_odd  = plt.subplot(2, 1, 1)
					w['deinterXXodd'] = np.rot90(w['deinterXXodd'])
					ax_deinter_odd.imshow(w['deinterXXodd'],extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)
					self.deinterXXodd_data = w['deinterXXodd']
				
					ax_deinter_even = plt.subplot(2, 1, 2)
					w['deinterXXeven'] = np.rot90(w['deinterXXeven'])
					ax_deinter_even.imshow(w['deinterXXeven'],extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)
					self.deinterXXeven_data = w['deinterXXeven']

				else:
	# 				norm = mpl.colors.PowerNorm(gamma=.2)
	# 				norm.autoscale(self.XX)
					self.im = ax.imshow(XX,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation, norm=w['imshow_norm'],clim=clim)

	# 				self.im.set_clim(self.autoColorScale(XX.flatten()))

				if 'flipaxes' in style:
					ax.set_xlabel(coord_keys[-1])
					ax.set_ylabel(coord_keys[-2])
				else:
					ax.set_xlabel(coord_keys[-2])
					ax.set_ylabel(coord_keys[-1])
							

				title = ''
				for i in uniques_col_str:
					title = '\n'.join([title, '{:s}: {:g} (mV)'.format(i,getattr(data_byuniques,i).iloc[0])])
				print(title)
				if 'notitle' not in style:
					ax.set_title(title)
				# create an axes on the right side of ax. The width of cax will be 5%
				# of ax and the padding between cax and ax will be fixed at 0.05 inch.
				if drawCbar:
					from mpl_toolkits.axes_grid.inset_locator import inset_axes
					if cbar_location == 'inset':
						if cbar_orientation == 'horizontal':
							cax = inset_axes(ax,width='30%',height='10%',loc=2,borderpad=1)
						else: 
							cax = inset_axes(ax,width='30%',height='10%',loc=1)						
					else:
						divider = make_axes_locatable(ax)
						if cbar_orientation == 'horizontal':
							cax = divider.append_axes("top", size="10%", pad=0.05)
						else:
							cax = divider.append_axes("right", size="2.5%", pad=0.05)
						pos = list(ax.get_position().bounds)
					if hasattr(self, 'im'):
						self.cbar = plt.colorbar(self.im, cax=cax,orientation=cbar_orientation)
						cbar = self.cbar
					
						if cbar_orientation == 'horizontal':
							cbar.set_label(cbar_title,labelpad=-20, x=1.35)
	# 						cbar.ax.xaxis.set_label_position('top')
	# 						cbar.ax.yaxis.set_label_position('left')
						else:
							cbar.set_label(cbar_title)#,labelpad=-19, x=1.32)
				cnt+=1 #counter for subplots

		self.toggleFiddle()
		self.toggleLinedraw()
		self.toggleLinecut()
		return self.fig
	
	def plot2d(self,fiddle=False,n_index=None,value_axis = -1,style=['normal'],**kwargs): #previously scanplot
		# scanplot(file,fig=None,n_index=None,style=[],data=None,**kwargs):
		# kwargs go into matplotlib/pyplot plot command
		
		if not self.fig:
			self.fig = plt.figure()
		
		if self.data is None:
			self.loadFile(file)
	
		uniques_col = []
	
		
		coord_keys = self.get_coordkeys()
		value_keys = self.get_valuekeys()
		#assume 2d plots with data in the two last columns
		uniques_col_str = coord_keys[:-1]
		
		
		uniques_axis_designations = []
		#do some filtering of the colstr to get seperate name and unit of said name
		for a in uniques_col_str:
			uniques_axis_designations.append(parseUnitAndNameFromColumnName(a))
				
		
		for i in uniques_col_str:
			col = getattr(self.data,i)
			uniques_col.append(col)

		if n_index is not None:
			n_index = np.array(n_index)
			n_subplots = len(n_index)
		
		for i,j in enumerate(buildLogicals(uniques_col)):
			if n_index is not None:
					if i not in n_index:
						continue
			filtereddata = self.data.loc[j]
			title =''
			for i,z in enumerate(uniques_col_str):
				title = '\n'.join([title, '{:s}: {:g} (mV)'.format(uniques_axis_designations[i],getattr(filtereddata,z).iloc[0])])

			x =  np.array(filtereddata.loc[:,coord_keys[-1]])
			y =  np.array(filtereddata.loc[:,value_keys[value_axis]])
	
			wrap = tstyle.getPopulatedWrap(style)
			wrap['XX'] = y
			tstyle.processStyle(style,wrap)
			ax = plt.plot(x,wrap['XX'],label=title,**kwargs)
				
		
		plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
			   ncol=2, mode="expand", borderaxespad=0.)
		ax = self.fig.axes[0]
		xaxislabel = parseUnitAndNameFromColumnName(coord_keys[-1])
		yaxislabel = parseUnitAndNameFromColumnName(value_keys[value_axis])
		
		ax.set_xlabel(xaxislabel[0]+'(' + xaxislabel[1] +')')
		ax.set_ylabel(yaxislabel[0]+'(' + yaxislabel[1] +')')
		return self.fig
		
	def get_coordkeys(self):
		coord_keys = [i['name'] for i in self.header if i['type']=='coordinate' ]
		return coord_keys
	def get_valuekeys(self):
		value_keys = [i['name'] for i in self.header if i['type']=='value' ]
		return value_keys
	def starplot(self,style=[]):
		if not self.fig:
			self.fig = plt.figure()
		
		if self.data is None:
			self.loadFile(file)
			
		data=self.data
		coordkeys = self.get_coordkeys()
		valuekeys = self.get_valuekeys()
		
		coordkeys_notempty=[k for k in coordkeys if len(data[k].unique()) > 1]
		n_subplots = len(coordkeys_notempty)
		width = 2
		import matplotlib.gridspec as gridspec
		gs = gridspec.GridSpec(int(n_subplots/width)+n_subplots%width, width)

		
		for n,k in enumerate(coordkeys_notempty):
			ax = plt.subplot(gs[n])
			for v in valuekeys:
				y= data[v]
				
				wrap = tstyle.getPopulatedWrap(style)
				wrap['XX'] = y
				tstyle.processStyle(style,wrap)
				
				ax.plot(data[k], wrap['XX'])
			ax.set_title(k)
		return self.fig

	def guessStyle(self):
		style=[]
		#autodeinterlace function
		#	if y[yu-1]==y[yu]: style.append('deinterlace0')

		#autodidv function
		y=self.filterdata.iloc[:,-2]
 		if (max(y) == -1*min(y) and max(y) <= 150):
 			style.extend(['mov_avg(m=1,n=10)','didv','mov_avg(m=1,n=5)','abs'])
			
		#default style is 'log'
		style.append = ['log']

		return style
		
	def sortdata(self,refresh=False):
		#some light caching
		if ((self.filterdata) is None) or refresh:
			cols = self.data.columns.tolist()
			self.filterdata = self.data.sort(cols[:-1])
			self.filterdata = self.filterdata.dropna(how='any')		
		return self.filterdata
		
	def getnDimFromData(self):
		#returns the 
		dims = np.array(self.getdimFromData())
		nDim = len(dims[dims > 1])
		return nDim
	
	def getdimFromData(self):
		#returns an array with the amount of unique values of each coordinate column
		
		dims = np.array([],dtype='int')
		filterdata = self.sortdata()
		#first determine the columns belong to the axes (not measure) coordinates
		cols = [i for i in self.header if (i['type'] == 'coordinate')]
		
		for i in cols:
			col = getattr(filterdata,i['name'])
			dims = np.hstack( ( dims ,len(col.unique())  ) )
		return dims
	
	def loadFile(self,file):

		self.header,self.skiprows = self.parseheader(file)
		self.data = pd.read_csv(file, sep='\t', comment='#',skiprows=self.skiprows,names=[i['name'] for i in self.header])
		self.data.name = file
		return self.data
	def toggleLinedraw(self):
		self.linedraw=Linedraw(self.fig)

		self.fig.drawbutton = toggleButton('draw', self.linedraw.connect)
		topwidget = self.fig.canvas.topLevelWidget()
		toolbar = topwidget.children()[2]
		action = toolbar.addWidget(self.fig.drawbutton)
		
		self.fig.linedraw = self.linedraw
	def toggleLinecut(self):
		self.linecut=Linecut(self.fig,self)

		self.fig.cutbutton = toggleButton('cut', self.linecut.connect)
		topwidget = self.fig.canvas.topLevelWidget()
		toolbar = topwidget.children()[2]
		action = toolbar.addWidget(self.fig.cutbutton)
		
		self.fig.linecut = self.linecut
		
	def toggleFiddle(self):
		from IPython.core import display
		
		if (mpl.get_backend() == 'Qt4Agg' or 'nbAgg'):
			self.fiddle = Fiddle(self.fig)

			self.fig.fiddlebutton = toggleButton('fiddle', self.fiddle.connect)
			topwidget = self.fig.canvas.topLevelWidget()
			toolbar = topwidget.children()[2]
			action = toolbar.addWidget(self.fig.fiddlebutton)

			#attach to the relevant figure to make sure the object does not go out of scope
			self.fig.fiddle = self.fiddle

	
	def toggleControls(self,state=None):
		self.bControls = not self.bControls
		if state == None:
			toggleFiddle()
		
	def opener(self,filename):
		import gzip
		f = open(filename,'rb')
		if (f.read(2) == '\x1f\x8b'):
			f.seek(0)
			return gzip.GzipFile(fileobj=f)
		else:
			f.seek(0)
			return f

	def parseheader(self,file):
		skipindex = 0
		
		with self.opener(file) as f:	
			for i, line in enumerate(f):
				if i < 3:
					continue
				if i > 5:
					if line[0] != '#': #find the skiprows accounting for the first linebreak in the header
						skipindex = i
						break
				if i > 300:
					break
		#nog een keer dunnetjes overdoen met read()
		f = self.opener(file)
		alltext= f.read(skipindex)		
		with self.opener(file) as myfile:
			alltext = [next(myfile) for x in xrange(skipindex)]
		alltext= ''.join(alltext)
	
		#doregex
		coord_expression = re.compile(r"""
									^\#\s*Column\s(.*?)\:
									[\r\n]{0,2}
									\#\s*end\:\s(.*?)
									[\r\n]{0,2}
									\#\s*name\:\s(.*?)
									[\r\n]{0,2}
									\#\s*size\:\s(.*?)
									[\r\n]{0,2}
									\#\s*start\:\s(.*?)
									[\r\n]{0,2}
									\#\s*type\:\s(.*?)[\r\n]{0,2}$ 
									
									"""#annoying \r's...
									,re.VERBOSE |re.MULTILINE)
		
		val_expression = re.compile(r"""
									^\#\s*Column\s(.*?)\:
									[\r\n]{0,2}
									\#\s*name\:\s(.*?)
									[\r\n]{0,2}
									\#\s*type\:\s(.*?)[\r\n]{0,2}$
									"""
									,re.VERBOSE |re.MULTILINE)
		coord=  coord_expression.findall(alltext) 
		val  = val_expression.findall(alltext)
		coord = [ zip(('column','end','name','size','start','type'),x) for x in coord]
		coord = [dict(x) for x in coord]
		val = [ zip(('column','name','type'),x) for x in val]
		val = [dict(x) for x in val]
		
		header=coord+val
		self.skipindex = skipindex
		self.header = header
		
		return np.array(header),skipindex

	
	def exportToMtx(self):

		for j, i in enumerate(self.exportData):

			data = i
			print(j)
			m = self.exportDataMeta[j]

			sz = np.shape(data)
			#write
			try:
				fid = open('{:s}{:d}{:s}'.format(self.name, j, '.mtx'),'w+')
			except Exception as e:
				print('Couldnt create file: {:s}'.format(str(e)))
				return

			#example of first two lines
			#Units, Data Value at Z = 0.5 ,X, 0.000000e+000, 1.200000e+003,Y, 0.000000e+000, 7.000000e+002,Nothing, 0, 1
			#850 400 1 8
			str1 = 'Units, Name: {:s}, {:s}, {:f}, {:f}, {:s}, {:f}, {:f}, {:s}, {:f}, {:f}\n'.format(
				m['datasetname'],
				m['xname'],
				m['xlims'][0],
				m['xlims'][1],
				m['yname'],
				m['ylims'][0],
				m['ylims'][1],
				m['zname'],
				m['zlims'][0],
				m['zlims'][1]
				)
			floatsize = 8
			str2 = '{:d} {:d} {:d} {:d}\n'.format(m['xu'],m['yu'],1,floatsize)
			fid.write(str1)
			fid.write(str2)
			#reshaped = np.reshape(data,sz[0]*sz[1],1)
			data.tofile(fid)
			fid.close()

def opener(filename):
	import gzip
	f = open(filename,'rb')
	if (f.read(2) == '\x1f\x8b'):
		f.seek(0)
		return gzip.GzipFile(fileobj=f)
	else:
		f.seek(0)
		return f

