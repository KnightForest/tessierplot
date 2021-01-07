
#tessier.py
#tools for plotting all kinds of files, with fiddle control etc

##Only load this part on first import, calling this on reload has dire consequences
## Note: there is still a bug where closing a previously plotted window and plotting another plot causes the window and the kernel to hang

try:
	from PyQt5 import QtCore
except:
	isqt5 = False
	try:
		from PyQt4 import QtCore
	except:
		isqt4 = False
	else:
		isqt4 = True
else:
	isqt5=True

import IPython
ipy=IPython.get_ipython()
if isqt5:
	ipy.magic("pylab qt5")
	qtaggregator = 'Qt5Agg'
elif isqt4:
	ipy.magic("pylab qt")
	qtaggregator = 'Qt4Agg'
else:
	print('no backend found.')

import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.signal import argrelmax
from scipy.interpolate import griddata

import numpy as np
import math
import re
import matplotlib.ticker as ticker
import pandas as pd

#all tessier related imports
from .gui import *
from . import styles
from .data import Data
from . import helpers
from . import colorbar

_plot_width = 7. # in inch (ffing inches eh)
_plot_height = 7. # in inch

_plot_width_thumb = 4. # in inch (ffing inches eh)
_plot_height_thumb = 4. # in inch

_fontsize_plot_title = 10
_fontsize_axis_labels = 10
_fontsize_axis_tick_labels = 10

_fontsize_plot_title = 10
_fontsize_axis_labels = 10
_fontsize_axis_tick_labels = 10

rcP = {	 'figure.figsize': (_plot_width, _plot_height), #(width in inch, height in inch)
		'axes.labelsize':  _fontsize_axis_labels,
		'xtick.labelsize': _fontsize_axis_tick_labels,
		'ytick.labelsize': _fontsize_axis_tick_labels,
		'legend.fontsize': 5.,
		'backend':qtaggregator
		}

rcP_thumb = {  'figure.figsize': (_plot_width_thumb, _plot_height_thumb), #(width in inch, height in inch)
		'axes.labelsize':  _fontsize_axis_labels,
		'xtick.labelsize': _fontsize_axis_tick_labels,
		'ytick.labelsize': _fontsize_axis_tick_labels,
		'legend.fontsize': 5.,
		'backend':qtaggregator
		}
		
def parseUnitAndNameFromColumnName(inp):
	reg = re.compile(r'\{(.*?)\}')
	z = reg.findall(inp)
	if not z: # if names don't follow the convention, just use what you get
		z = inp
	return z


def loadCustomColormap(file=helpers.get_asset('cube2.txt')):
	do = np.loadtxt(file)
	ccmap = mpl.colors.LinearSegmentedColormap.from_list('name',do)

	ccmap.set_under(do[0])
	ccmap.set_over(do[-1])
	return ccmap

class plotR(object):
	def __init__(self,file,isthumbnail=False,thumbs = None):
		self.fig = None
		self.file = file
		self.isthumbnail = isthumbnail
		if (thumbs is not None):
			self.thumbfile = thumbs[0]
			self.thumbfile_datadir = thumbs[1]

		self.data  = Data.from_file(filepath=file)
		self.name  = file
		self.exportData = []
		self.exportDataMeta = []
		self.bControls = True #boolean controlling state of plot manipulation buttons
		self.valueaxes_n = 0
		#print(self.data._header)
		#print(self.data.coordkeys)
		
	def is2d(self,**kwargs):
		nDim = self.data.ndim_sparse
		#if the uniques of a dimension is less than x, plot in sequential 2d, otherwise 3d

		#maybe put logic here to plot some uniques as well from nonsequential axes?
		filter = self.data.dims < 5
		filter_neg = np.array([not x for x in filter],dtype="bool")

		coords = np.array(self.data.coordkeys)

		return len(coords[filter_neg]) < 2

	def measurementorder(self,**kwargs):
		return len(np.array(self.data.coordkeys))

	# def quickplot_processed(self,**kwargs):
		# coords = np.array(self.data.coordkeys)
		# filter = self.data.dims < 5

		# uniques_col_str = coords[filter]
		# try:
			# if self.isthumbnail:
				# for k in rcP:
					# mpl.rcParams[k] = rcP_thumb[k]
			# else:
				# for k in rcP:
					# mpl.rcParams[k] = rcP[k]
			# if self.is2d():
				# fig = self.plot2d(uniques_col_str=uniques_col_str,**kwargs)
			# else:
				# fig = self.plot3d(uniques_col_str=uniques_col_str,**kwargs)
				# self.exportToMtx()
			# if self.isthumbnail:
				# fig.savefig(self.thumbfile,bbox_inches='tight' )
				# fig.savefig(self.thumbfile_datadir,bbox_inches='tight' )
				# plt.close(fig)
		# except Exception as e:
			# print('fail in quickplot')
			# print(e)
		
		# return fig

	def quickplot(self,**kwargs):
		coords = np.array(self.data.coordkeys)
		filter = self.data.dims < 5

		uniques_col_str = coords[filter]
		try:
			if self.isthumbnail:
				for k in rcP:
					mpl.rcParams[k] = rcP_thumb[k]
			# else:
			#	for k in rcP:
			#		mpl.rcParams[k] = rcP[k]
			if self.is2d():
				fig = self.plot2d(uniques_col_str=uniques_col_str,**kwargs)
			else:
				fig = self.plot3d(uniques_col_str=uniques_col_str,**kwargs)
				#self.exportToMtx()
			if self.isthumbnail:
				#print(self.valueaxes_n)
				fig.set_size_inches(_plot_width_thumb, (self.valueaxes_n-1)*2+_plot_height_thumb)
				fig.savefig(self.thumbfile,bbox_inches='tight' )
				fig.savefig(self.thumbfile_datadir,bbox_inches='tight' )
				plt.close(fig)
		except Exception as e:
			print('fail in quickplot')
			print(e)
		
		return fig

	def autoColorScale(self,data):
		#filter out NaNs or infinities, should any have crept in
		data = data[np.isfinite(data)]
		m=2
		datastd= data[abs(data - np.mean(data)) < m * np.std(data)]
		values, edges = np.histogram(datastd, 256)
		maxima = edges[argrelmax(values,order=24)]
		try:
			if maxima.size>0:
				cminlim , cmaxlim = maxima[0] , np.max(datastd)
			else:
				cminlim , cmaxlim = np.min(datastd) , np.max(datastd)
		except Exception as e:
			print('Colorscale fallback to min-max value')
			cminlim , cmaxlim = np.min(data) , np.max(data)
		return (cminlim,cmaxlim)

	def plothigherorder(self,	 massage_func=None,
						uniques_col_str=[],
						drawCbar=True,
						cax_destination=None,
						subplots_args={'top':0.96, 'bottom':0.17, 'left':0.14, 'right':0.85,'hspace':0.4},
						ax_destination=None,
						n_index=None,
						style=['normal'],
						xlims_manual=None,
						ylims_manual=None,
						clim=None,
						aspect='auto',
						interpolation='nearest',
						value_axis=-1,
						sweepoverride=False,
						imshow=True,
						cbar_orientation='vertical',
						cbar_location ='normal',
						filter_raw = True,
						ccmap = None,
						#supress_plot = False, #Added suppression of plot option
						norm = 'nan', #Added for NaN value support
						**kwargs):
		# Placeholder for plotting cube maps and higher order measurement. Should generate a loop for plotting colormaps.
		print('Cubeplotplaceholder')
		pass

	def plot3d(self,	massage_func=None,
						uniques_col_str=[],
						drawCbar=True,
						cax_destination=None,
						subplots_args={'top':0.96, 'bottom':0.17, 'left':0.14, 'right':0.85,'hspace':0.4},
						ax_destination=None,
						n_index=None,
						style=['normal'],
						xlims_manual=None,
						ylims_manual=None,
						clim=None,
						aspect='auto',
						interpolation='nearest',
						value_axis=-1,
						sweepoverride=False,
						imshow=True,
						cbar_orientation='vertical',
						cbar_location ='normal',
						filter_raw = True,
						ccmap = None,
						supress_plot = False, #Added suppression of plot option
						norm = 'nan', #Added for NaN value support
						axislabeltype = 'label', #Use 'label' or 'name' on axis labels
						**kwargs):
		#some housekeeping
		if not self.fig and not ax_destination:
			self.fig = plt.figure()
			self.fig.subplots_adjust(**subplots_args)
			
		if not ccmap:
			self.ccmap = loadCustomColormap()
		else:
			self.ccmap = ccmap
		
		if len(uniques_col_str)==0:
			coords = np.array(self.data.coordkeys)
			filter = self.data.dims < 2
			uniques_col_str = coords[filter]

		#determine how many subplots we need
		n_subplots = 1

		#make a list of uniques per column associated with column name
		coord_keys,coord_units,coord_labels = self.data.coordkeys_n
		value_keys,value_units,value_labels = self.data.valuekeys_n
		coord_keys_raw,coord_units_raw,value_labels_raw = self.data.coordkeys_n

		#Filtering raw value axes
		if filter_raw== True:
			value_keys_filtered = []
			value_labels_filtered = []
			value_units_filtered = []
			for n,value_label in enumerate(value_labels):
				if value_label.find('raw')==-1 and value_label.find('Raw')== -1:
					value_labels_filtered.append(value_label)
					value_keys_filtered.append(value_keys[n])
					value_units_filtered.append(value_units[n])
			value_keys = value_keys_filtered
			value_units = value_units_filtered
			value_labels = value_labels_filtered

		#make a list of uniques per column associated with column name
		uniques_by_column = dict(zip(coord_keys + value_keys, self.data.dims))

		#by this array
		for i in uniques_col_str:
			n_subplots *= uniques_by_column[i]

		if n_index is not None:
			n_index = np.array(n_index)
			n_subplots = len(n_index)

		if n_subplots > 1:
			width = 1
		else:
			width = 1
		n_valueaxes = len(value_keys)	
		
		if value_axis == -1:
			value_axes = range(n_valueaxes)
		else:
			value_axes = list([value_axis])
		self.valueaxes_n = len(value_axes) 
		width = 1#len(value_axes)
		height = len(value_axes)
		n_subplots = n_subplots *width#int(n_subplots/width)+n_subplots%width
		gs = gridspec.GridSpec(height,width)
		cnt=0 #subplot counter

		#enumerate over the generated list of unique values specified in the uniques columns
		for j,ind in enumerate(self.data.make_filter_from_uniques_in_columns(uniques_col_str)):
			#each value axis needs a plot

			for value_axis in value_axes:
				#plot only if number of the plot is indicated
				if n_index is not None:
					if j not in n_index:
						continue
				us=uniques_col_str
				coord_keys = [key for key in coord_keys if key not in uniques_col_str ]
				coord_units = list(coord_units[i] for i in [i for i, key in enumerate(coord_keys) if key not in uniques_col_str])
				coord_labels = list(coord_labels[i] for i in [i for i, key in enumerate(coord_keys) if key not in uniques_col_str])

				data_slice = (self.data.loc[ind]).dropna(subset=[coord_keys[-2], coord_keys[-1]]) #Dropping rows with NaNs in coordinates
				xu = np.size(data_slice[coord_keys[-2]].unique())
				yu = np.size(data_slice[coord_keys[-1]].unique())
				lenz = np.size(data_slice[value_keys[value_axis]])
				print('xu: {:d}, yu: {:d}, lenz: {:d}'.format(xu,yu,lenz))
				

				if xu*yu > lenz: #Condition for unfinished measurement sweep
					
					missingpoints = xu*yu-lenz		
					xarr=np.full((missingpoints),data_slice[coord_keys[-2]].iloc[-1])
					# Determining the y-array, can bug when strange steps are used during measurement
					ystep1 = data_slice[coord_keys[-1]].iloc[-1]-data_slice[coord_keys[-1]].iloc[-2]
					ystep2 = data_slice[coord_keys[-1]].iloc[-2]-data_slice[coord_keys[-1]].iloc[-3]
					if np.abs(ystep1) < abs(ystep2):
						ystep=ystep1
					else:
						ystep=ystep2
					#ystep = np.float(np.format_float_scientific(ystep, unique=False, precision=10))
					ystart = data_slice[coord_keys[-1]].iloc[-1]+ystep
					yend = missingpoints*ystep+ystart-ystep
					yarr = np.linspace(ystart,yend,missingpoints)
					#print(ystep1,ystep2,ystep,ystart,yend,yarr)
					#for i in range(0,len(yarr)):
					#	yarr[i] = np.float(np.format_float_scientific(yarr[i], unique=False, precision=10))
					#print('yarr',yarr)
					zarr = np.zeros(int(xu*yu-lenz)) + np.nan
					concatdf = pd.DataFrame({coord_keys[-2]: xarr,
									   coord_keys[-1]: yarr,
									   value_keys[value_axis]: zarr})
					newdf = [data_slice,concatdf]
					data_slice = pd.concat(newdf, ignore_index=True)
					lenz = np.size(data_slice[value_keys[value_axis]])
					print('xu: {:d}, yu: {:d}, lenz: {:d} after adding nan for incomplete sweep'.format(xu,yu,lenz))

				#get the columns /not/ corresponding to uniques_cols
				#filter out the keys corresponding to unique value columns

				#now find out if there are multiple value axes
				#self.data_slice_unsorted = data_slice
				data_slice = (data_slice.sort_values(by=[coord_keys[-2],coord_keys[-1]]))#.reset_index(drop=True)
				self.data_slice = data_slice

				x=data_slice.loc[:,coord_keys[-2]]
				y=data_slice.loc[:,coord_keys[-1]]
				z=data_slice.loc[:,value_keys[value_axis]]

				XX = z.values.reshape(xu,yu)
				X = x.values.reshape(xu,yu)
				Y = y.values.reshape(xu,yu)

				#if hasattr(self, 'XX_processed'):
				#	XX = self.XX_processed

				self.x = x
				self.y = y
				self.z = z
				#now set the lims
				xlims = (x.min(),x.max())
				ylims = (y.min(),y.max())
				xnew = xlims
				ynew = ylims

				if not (xlims_manual == None):
					xnew = xlims
					if xlims_manual[0] > xlims[0]:
						xnew[0] = xlims_manual[0]
					
					if xlims_manual[1] > xlims[1]:
						xnew[1] = xlims_manual[1]

				if not (ylims_manual == None):
					ynew = ylims
					if ylims_manual[0] > ylims[0]:
						ynew[0] = ylims_manual[0]
					
					if ylims_manual[1] > ylims[1]:
						ynew[1] = ylims_manual[1]

				ext = xlims+ylims
				self.extent = ext

				#Gridding and interpolating unevenly spaced data
				extx = abs(ext[1]-ext[0])
				xdx = np.diff(X, axis=0)
				xdxshape = xdx.shape
				#for i in range(0,int(xdxshape[0])): # Rounding to finite precision to find stepsize
				#	for j in range(0,int(xdxshape[1])):
				#		xdx[i,j]=np.format_float_scientific(xdx[i,j], unique=False, precision=10)
				minxstep = np.nanmin(abs(xdx[xdx > 1e-19])) # removing rounding errors from diff
				minxsteps = int(round(extx/minxstep,0))+1
				exty = abs(ext[3]-ext[2])
				ydy = np.diff(Y, axis=1)#.astype(float)
				ydyshape = ydy.shape
				#for i in range(0,ydyshape[0]):
				#		for j in range(0,ydyshape[1]):
				#				ydy[i,j]=np.format_float_scientific(ydy[i,j], unique=False, precision=10)
				minystep = np.nanmin(abs(ydy[ydy > 1e-19])) # removing rounding errors from diff
				minysteps = int(round(exty/minystep,0))+1
				if minysteps > 100*yu: #limiting the interpolation stepsize
					minysteps = 100*yu
				if minxsteps > xu or minysteps > yu:
					print('Unevenly spaced data detected, cubic interpolation will be performed. \n New dimension:', 1*minxsteps,1*minysteps)
					# grid_x, grid_y and points are divided by their respective stepsize in x and y to get a properly weighted interpolation
					grid_x, grid_y = np.mgrid[ext[0]:ext[1]:minxsteps*1j, ext[2]:ext[3]:minysteps*1j]
					gridxstep = np.abs(grid_x[1,0]-grid_x[0,0])
					gridystep = np.abs(grid_y[0,1]-grid_y[0,0])
					#print(gridxstep,gridystep)
					grid_x /= gridxstep
					grid_y /= gridystep
					points = np.transpose(np.array([x/gridxstep,y/gridystep]))
					z1=np.array(z)
					# Getting index for nans in points and values, they need to be removed for cubic interpolation to work.
					indexnonans=np.invert(np.isnan(points[:,0]))*np.invert(np.isnan(points[:,1]))*np.invert(np.isnan(z1))
					try:
						XX = griddata(np.stack((points[:,0][indexnonans],points[:,1][indexnonans]),axis=1), np.array(z)[indexnonans], (grid_x, grid_y), method='cubic')
					
					#	print(XX)
					except:
						print('Cubic interpolation failed, falling back to \'nearest\'')
						XX = griddata(points, np.array(z), (grid_x, grid_y), method='nearest')
					X = grid_x
					Y = grid_y
				self.XX = XX
				self.X = X
				self.Y = Y

				#determine stepsize for di/dv, inprincipe only y step is used (ie. the diff is also taken in this direction and the measurement swept..)
				xstep = float(xlims[1] - xlims[0])/(len(self.X[:,0])-1)
				ystep = float(ylims[1] - ylims[0])/(len(self.Y[0,:])-1)
				self.exportData.append(XX)
				try:
					m={
						'xu':xu,
						'yu':yu,
						'xlims':xlims,
						'ylims':ylims,
						'zlims':(0,0),
						'xname':coord_keys[-2],
						'yname':coord_keys[-1],
						'zname':'unused',
						'datasetname':self.name}
					self.exportDataMeta = np.append(self.exportDataMeta,m)
				except Exception as e:
					print(e)
					pass
				if ax_destination is None:
					ax = plt.subplot(gs[cnt])
				else:
					ax = ax_destination
				cbar_title = ''

				if type(style) != list:
					style = list([style])

				if axislabeltype == 'label':
					xaxislabel = coord_labels[-2] 
					yaxislabel = coord_labels[-1]
					cbar_quantity = value_labels[value_axis] 
				elif axislabeltype == 'name':
					xaxislabel = coord_keys[-2] 
					yaxislabel = coord_keys[-1]
					cbar_quantity = value_keys[value_axis] 
				else:
					raise Exception('Wrong axislabeltype argument, must be \'name\' or \'label\'')
				xaxisunit = coord_units[-2]
				yaxisunit = coord_units[-1]
				cbar_unit = value_units[value_axis]
				#wrap all needed arguments in a datastructure
				sbuffer = ''
				cbar_trans = [] #trascendental tracer :P For keeping track of logs and stuff
				w = styles.getPopulatedWrap(style)
				w2 = {
						'ext':ext,
						'xstep': xstep,
						'ystep': ystep,
						'XX': XX,
						'X': X,
						'Y': Y,
						'x': x,
						'y': y,
						'z': z,
						'cbar_quantity': cbar_quantity, 
						'cbar_unit': cbar_unit, 
						'cbar_trans':cbar_trans, 
						'buffer':sbuffer, 
						'xlabel':xaxislabel, 
						'xunit':xaxisunit, 
						'ylabel':yaxislabel, 
						'yunit':yaxisunit}
				for k in w2:
					w[k] = w2[k]
				w['massage_func']=massage_func
				styles.processStyle(style, w)
				#unwrap
				ext = w['ext']
				XX = w['XX']
				cbar_trans_formatted = ''.join([''.join(s+'(') for s in w['cbar_trans']])
				cbar_title = cbar_trans_formatted + w['cbar_quantity'] + ' (' + w['cbar_unit'] + ')'
				if len(w['cbar_trans']) is not 0:
					cbar_title = cbar_title + ')'

				self.stylebuffer = w['buffer'] 
				self.xlabel = w['xlabel']
				self.xunit = w['xunit']
				self.ylabel= w['ylabel']
				self.yunit = w['yunit']
				self.XX_processed = XX

				if w['imshow_norm'] == None: # Support for plotting NaN values in a different color
					self.imshow_norm = colorbar.MultiPointNormalize()
				else:
					self.imshow_norm = w['imshow_norm']
				if norm == 'nan':
					self.imshow_norm = None

				# This deinterlace needs to be reworked. There are no colorbars for instance..
				if 'deinterlace' in style:
					self.fig = plt.figure()
					ax_deinter_odd	= plt.subplot(2, 1, 1)
					xx_odd = np.rot90(w['deinterXXodd'])
					ax_deinter_odd.imshow(xx_odd,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)
					self.deinterXXodd_data = xx_odd

					ax_deinter_even = plt.subplot(2, 1, 2)
					xx_even = np.rot90(w['deinterXXeven'])
					ax_deinter_even.imshow(xx_even,extent=ext, cmap=plt.get_cmap(self.ccmap),aspect=aspect,interpolation=interpolation)
					self.deinterXXeven_data = xx_even
				else:
					if imshow:
						#masked_array_nans = np.ma.array(np.rot90(XX), mask=np.isnan(np.rot90(XX)))
						#masked_array_nans = np.rot90(XX)
						colormap = (plt.get_cmap(self.ccmap))
						colormap.set_bad('grey',1.0)
						#self.im = ax.imshow(masked_array_nans,extent=ext, cmap=colormap,aspect=aspect,interpolation=interpolation, clim=clim)
						#self.im = ax.imshow(np.rot90(XX) ,extent=ext, cmap=plt.get_cmap(self.ccmap) ,aspect=aspect,interpolation=interpolation, clim=clim)
						self.im = ax.imshow(np.rot90(XX), extent=ext, cmap=plt.get_cmap(self.ccmap), aspect=aspect, interpolation=interpolation, norm=self.imshow_norm,clim=clim)
					else:
						xs = np.linspace(ext[0],ext[1],XX.shape[0])
						ys = np.linspace(ext[2],ext[3],XX.shape[1])
						xv,yv = np.meshgrid(xs,ys) 

						colormap = (plt.get_cmap(self.ccmap)) # Support for plotting NaN values
						colormap.set_bad('none',1.0)
						self.im = ax.pcolormesh(xv,yv,np.rot90(np.fliplr(XX)),cmap=plt.get_cmap(self.ccmap), vmin=clim[0], vmax=clim[1])
					if not clim:
						self.im.set_clim(self.autoColorScale(XX.flatten()))
				#ax.locator_params(nbins=5, axis='y') #Added to hardcode number of x ticks.
				#ax.locator_params(nbins=7, axis='x')
				if 'flipaxes' in style:
					xaxislabelwithunit = self.ylabel +	' (' + self.yunit + ')'
					yaxislabelwithunit = self.xlabel +	' (' + self.xunit + ')'
				else:
					xaxislabelwithunit = self.xlabel +	' (' + self.xunit + ')'
					yaxislabelwithunit = self.ylabel +	' (' + self.yunit + ')'
				ax.set_xlabel(xaxislabelwithunit)
				ax.set_ylabel(yaxislabelwithunit)
				
				title = ''
				for i in uniques_col_str:
					title = '\n'.join([title, '{:s}: {:g} {:s}'.format(i,getattr(data_slice,i).iloc[0], coord_units_raw[coord_keys_raw.index(i)] )])

				if 'notitle' not in style:
					ax.set_title(title)
				# create an axes on the right side of ax. The width of cax will be 5%
				# of ax and the padding between cax and ax will be fixed at 0.05 inch.
				if drawCbar:
					from mpl_toolkits.axes_grid1.inset_locator import inset_axes
					cax = None
					if cax_destination:
						cax = cax_destination
					elif cbar_location == 'inset':
						if cbar_orientation == 'horizontal':
							cax = inset_axes(ax,width='30%',height='10%',loc=2,borderpad=1)
						else:
							cax = inset_axes(ax,width='30%',height='10%',loc=1)
					else:
						divider = make_axes_locatable(ax)
						if cbar_orientation == 'horizontal': # Added some hardcode config for colorbar, more pretty out of the box
							cax = divider.append_axes("top", size="5%", pad=0.05)
							cax.set_aspect(0.1)
							cax.set_anchor('E')
						else:
							cax = divider.append_axes("right", size="2.5%", pad=0.05)
						pos = list(ax.get_position().bounds)
					if hasattr(self, 'im'):
						self.cbar = colorbar.create_colorbar(cax, self.im, orientation=cbar_orientation)
						cbar = self.cbar

						if cbar_orientation == 'horizontal': #Added some hardcode config for colorbar, more pretty out of the box
							cbar.set_label(cbar_title,labelpad=-15, x = -0.3, horizontalalignment='right')
							cbar.ax.xaxis.set_label_position('top')
							cbar.ax.xaxis.set_ticks_position('top')
							tick_locator = ticker.MaxNLocator(nbins=3)
							cbar.locator = tick_locator

						else:
							tick_locator = ticker.MaxNLocator(nbins=3)
							cbar.locator = tick_locator
							cbar.set_label(cbar_title)#,labelpad=-19, x=1.32)
							cbar.update_ticks()
						
						self.cbar = cbar
						cbar.update_ticks()
						if supress_plot == False:
							plt.show()
				self.ax = ax
				cnt+=1 #counter for subplots
		
		
		if self.fig and (mpl.get_backend() in [qtaggregator , 'nbAgg']):
			self.toggleFiddle()
			self.toggleLinedraw()
			self.toggleLinecut()
	
		return self.fig

	def plot2d(self,fiddle=False,
					n_index=None,
					value_axis = -1,
					style=['normal'],
					uniques_col_str=[],
					legend=False,
					ax_destination=None,
					subplots_args={'top':0.96, 'bottom':0.17, 'left':0.14, 'right':0.85,'hspace':0.3},
					massage_func=None,
					filter_raw=True, #Do not show plots with 'Raw' string in label
					axislabeltype = 'label', #Use 'label' or 'name' on axis labels
                    supress_plot = False,
					**kwargs):
					
		if not self.fig and not ax_destination:
			self.fig = plt.figure()
			self.fig.subplots_adjust(**subplots_args)

		# if len(uniques_col_str)==0:
		#	coords = np.array(self.data.coordkeys)
		#	filter = self.data.dims < 5
		#	uniques_col_str = coords[filter]

		#determine how many subplots we need
		n_subplots = 1
		coord_keys,coord_units,coord_labels = self.data.coordkeys_n
		value_keys,value_units,value_labels = self.data.valuekeys_n
		coord_keys_raw,coord_units_raw,coord_labels_raw = self.data.coordkeys_n
		#Filtering raw value axes
		if filter_raw== True:
			value_keys_filtered = []
			value_labels_filtered = []
			value_units_filtered = []
			for n,value_label in enumerate(value_labels):
				if value_label.find('raw')==-1 and value_label.find('Raw')== -1:
					value_labels_filtered.append(value_label)
					value_keys_filtered.append(value_keys[n])
					value_units_filtered.append(value_units[n])
			value_keys = value_keys_filtered
			value_units = value_units_filtered
			value_labels = value_labels_filtered
		#make a list of uniques per column associated with column name
		uniques_by_column = dict(zip(coord_keys + value_keys, self.data.dims))

		#assume 2d plots with data in the two last columns
		if len(uniques_col_str)==0:
			uniques_col_str = coord_keys[:-1]

		if n_index is not None:
			n_index = np.array(n_index)
			n_subplots = len(n_index)

		if n_subplots > 1:
			width = 2
		else:
			width = 1
		n_valueaxes = len(value_keys)
		if value_axis == -1:
			value_axes = range(n_valueaxes)
		else:
			if type(value_axis) is not list:
				value_axes = list([value_axis])
			else:
				value_axes = value_axis
		self.valueaxes_n = len(value_axes) 

		width = len(value_axes)
		n_subplots = n_subplots * width
		gs = gridspec.GridSpec(width,int(n_subplots/width)+n_subplots%width)

		if n_index is not None:
			n_index = np.array(n_index)
			n_subplots = len(n_index)
		ax = None
		for i,j in enumerate(self.data.make_filter_from_uniques_in_columns(uniques_col_str)):
		
			for k,value_axis in enumerate(value_axes):
				if n_index is not None:
						if i not in n_index:
							continue
				data = self.data.sorted_data[j]
				#filter out the keys corresponding to unique value columns
				us=uniques_col_str
				coord_keys = [key for key in coord_keys if key not in uniques_col_str]
				#now find out if there are multiple value axes
				#value_keys, value_units = self.data.valuekeys

				x=data.loc[:,coord_keys[-1]]
				xx=data.loc[:,value_keys[value_axis]]
				parser = self.data.determine_parser

				if axislabeltype == 'label' and len(coord_labels_raw) == len(coord_keys_raw):
					xaxislabel = coord_labels[-1]
				else: #else defaulting to column name for axis labels
					xaxislabel = coord_keys[-1] 
				if axislabeltype == 'label' and len(value_labels) == len(value_keys):
					cbar_quantity = value_labels[value_axis]
				else: #else defaulting to column name for axis labels
					cbar_quantity = value_keys[value_axis]

				xaxisunit = coord_units[-1]
				cbar_unit = value_units[value_axis]

				npx = np.array(x)
				npxx = np.array(xx)
				#print(npx,npy)
				if len(npx) > 1:
					xstep = float(abs(npx[-1] - npx[0]))/(len(npx)-1)
				else:
					xstep = np.nan
				#ystep = float(abs(npy[-1] - npy[0]))/(len(npy)-1)
				title =''

				for i,z in enumerate(uniques_col_str):
					pass
					# this crashes sometimes. did not investiagte yet what the problem is. switched off in the meantime
					#title = '\n'.join([title, '{:s}: {:g}'.format(uniques_axis_designations[i],data[z].iloc[0])])
				self.XX = npxx

				wrap = styles.getPopulatedWrap(style)
				wrap['XX'] = npxx
				wrap['X']  = npx
				wrap['Y'] = np.nan #not existing
				wrap['xlabel'] = xaxislabel
				wrap['xunit'] = xaxisunit
				wrap['cbar_quantity'] = cbar_quantity
				wrap['cbar_unit'] = cbar_unit
				wrap['massage_func'] = massage_func
				wrap['xstep'] = xstep
				wrap['ystep'] = np.nan
				styles.processStyle(style,wrap)
				xaxislabelwithunit = wrap['xlabel'] + ' (' + wrap['xunit'] + ')'
				yaxislabelwithunit = wrap['cbar_quantity'] + ' (' + wrap['cbar_unit'] + ')'

				self.stylebuffer = wrap['buffer'] 
				self.xaxislabel = wrap['xlabel']
				self.xaxisunit = wrap['xunit']
				self.yaxislabel= wrap['cbar_quantity']
				self.yaxisunit = wrap['cbar_unit']
				self.XX_processed = wrap['XX']
				self.X = wrap['X']
				if supress_plot == False:
					if ax_destination:
						ax = ax_destination
					else:
						ax = plt.subplot(gs[k])
					ax.plot(wrap['X'],wrap['XX'],'o-', fillstyle='none', markersize=2,label=title,**kwargs)
					#self.cutAx.plot(xx,z,'o-',fillstyle='none',markersize=2)
					if legend:
						plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
							   ncol=2, mode="expand", borderaxespad=0.)
					if ax:
						ax.set_xlabel(xaxislabelwithunit)
						ax.set_ylabel(yaxislabelwithunit)
		
		return self.fig


	def starplot(self,style=[]):
		if not self.fig:
			self.fig = plt.figure()

		data=self.data
		coordkeys = self.data.coordkeys
		valuekeys = self.data.valuekeys

		coordkeys_notempty=[k for k in coordkeys if len(data[k].unique()) > 1]
		n_subplots = len(coordkeys_notempty)
		width = 2
		import matplotlib.gridspec as gridspec
		gs = gridspec.GridSpec(int(n_subplots/width)+n_subplots%width, width)


		for n,k in enumerate(coordkeys_notempty):
			ax = plt.subplot(gs[n])
			for v in valuekeys:
				y= data[v]

				wrap = styles.getPopulatedWrap(style)
				wrap['XX'] = y
				styles.processStyle(style,wrap)

				ax.plot(data[k], wrap['XX'])
			ax.set_title(k)
		return self.fig

	def guessStyle(self):
		#Make guesstyle dependent on unit, i.e., A or V should become derivatives.
		style=[]
		#autodeinterlace function
		#	if y[yu-1]==y[yu]: style.append('deinterlace0')

		#autodidv function
		y=self.data.sorted_data.iloc[:,-2]
		if (max(y) <= 15000):
			style.extend(['mov_avg(m=1,n=3)','didv','mov_avg(m=1,n=3)'])

		#default style is 'log'
		#style.append('fixlabels')
		return style

	def toggleLinedraw(self):
		self.linedraw=Linedraw(self.fig)

		self.fig.drawbutton = toggleButton('draw', self.linedraw.connect)
		topwidget = self.fig.canvas.window()
		toolbar = topwidget.children()[1]
		action = toolbar.addWidget(self.fig.drawbutton)

		self.fig.linedraw = self.linedraw
	def toggleLinecut(self):
		self.linecut=Linecut(self.fig,self)

		self.fig.cutbutton = toggleButton('cut', self.linecut.connect)
		topwidget = self.fig.canvas.window()
		toolbar = topwidget.children()[1]
		action = toolbar.addWidget(self.fig.cutbutton)

		self.fig.linecut = self.linecut

	def toggleFiddle(self):
		from IPython.core import display

		self.fiddle = Fiddle(self.fig)
		self.fig.fiddlebutton = toggleButton('fiddle', self.fiddle.connect)
		topwidget = self.fig.canvas.window()
		toolbar = topwidget.children()[1]
		action = toolbar.addWidget(self.fig.fiddlebutton)

		#attach to the relevant figure to make sure the object does not go out of scope
		self.fig.fiddle = self.fiddle
	
	def exportToMtx(self):

		for j, i in enumerate(self.exportData):

			data = i
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