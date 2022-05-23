
#tessier.py
#tools for plotting all kinds of files, with fiddle control etc

##Only load this part on first import, calling this on reload has dire consequences
## Note: there is still a bug where closing a previously plotted window and plotting another plot causes the window and the kernel to hang

try: # Importing PyQt5 if available
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

import IPython #Importing IPython modules
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
from quantiphy import Quantity

import numpy as np
import math
import re
import matplotlib.ticker as ticker
import pandas as pd
import copy

#all tessier related imports
from .gui import *
from . import styles
from .data import Data
from . import helpers
from . import colorbar

# Setting default plot properties
_plot_width = 4.75 # in inch (ffing inches eh)
_plot_height = 4.25 # in inch

_plot_width_thumb = 4.1 # in inch (ffing inches eh)
_plot_height_thumb = 3.25 # in inch

_fontsize_plot_title = 10
_fontsize_axis_labels = 10
_fontsize_axis_tick_labels = 10

_fontsize_plot_title_thumb = 9
_fontsize_axis_labels_thumb = 9
_fontsize_axis_tick_labels_thumb = 9

# Settings for 'normal' plots
rcP = {	 'figure.figsize': (_plot_width, _plot_height), #(width in inch, height in inch)
		'axes.labelsize':  _fontsize_axis_labels,
		'xtick.labelsize': _fontsize_axis_tick_labels,
		'ytick.labelsize': _fontsize_axis_tick_labels,
		'legend.fontsize': 5.,
		'backend':qtaggregator
		}

# Settings for generating thumbnails
rcP_thumb = {  'figure.figsize': (_plot_width_thumb, _plot_height_thumb), #(width in inch, height in inch)
		'axes.labelsize':  _fontsize_axis_labels_thumb,
		'xtick.labelsize': _fontsize_axis_tick_labels_thumb,
		'ytick.labelsize': _fontsize_axis_tick_labels_thumb,
		'legend.fontsize': 5.,
		'backend':qtaggregator
		}
		
# Parse regex for old dat files to extract units and names
def parseUnitAndNameFromColumnName(inp):
	reg = re.compile(r'\{(.*?)\}')
	z = reg.findall(inp)
	if not z: # if names don't follow the convention, just use what you get
		z = inp
	return z

# Loading of custom colormap from file
def loadCustomColormap(file=helpers.get_asset('cube2.txt')):
	do = np.loadtxt(file)
	ccmap = mpl.colors.LinearSegmentedColormap.from_list('name',do)

	ccmap.set_under(do[0])
	ccmap.set_over(do[-1])
	return ccmap

# Main plotting class
class plotR(object):
	def __init__(self,file,isthumbnail=False,thumbs = None):
		self.fig = None
		self.file = file
		self.isthumbnail = isthumbnail
		if (thumbs != None):
			self.thumbfile = thumbs[0]
			self.thumbfile_datadir = thumbs[1]

		self.data  = Data.from_file(filepath=file)
		self.name  = file
		self.exportData = []
		self.exportDataMeta = []
		self.bControls = True #boolean controlling state of plot manipulation buttons
		self.quantiphy_ignorelist = ['dBm','dB', 'rad', 'rad', 'Deg', 'deg', 'Arb', 'arb.', 'nA', 'mV']
		#print(self.data._header)
		#print(self.data.coordkeys)
		
	# Checks wheter plot is 2d and return boolean
	def is2d(self,**kwargs):
		nDim = self.data.ndim_sparse
		#if the uniques of a dimension is less than x, plot in sequential 2d, otherwise 3d

		#maybe put logic here to plot some uniques as well from nonsequential axes?
		filter = self.data.dims < 2
		filter_neg = np.array([not x for x in filter],dtype="bool")

		coords = np.array(self.data.coordkeys)

		return len(coords[filter_neg]) < 2

	# Function that calls either plot2d (lines) or plot3d (colorplot) based on number of coordinate axes and uniques
	def quickplot(self,**kwargs):
		coords = np.array(self.data.coordkeys)
		filter = self.data.dims < 2 # kick out coordinate axes with less than 2 entries

		uniques_col_str = coords[filter]
		try:
			if self.isthumbnail: # If generating thumbnails, different arguments can be given to plot2d or plot3d
				if self.is2d():
					fig = self.plot2d(uniques_col_str=uniques_col_str,**kwargs)
				else:
					fig = self.plot3d(uniques_col_str=uniques_col_str,cbar_orientation='vertical',**kwargs)
				fig.savefig(self.thumbfile,bbox_inches='tight' )
				fig.savefig(self.thumbfile_datadir,bbox_inches='tight' )
				plt.close(fig)
			else:
				if self.is2d():
					fig = self.plot2d(uniques_col_str=uniques_col_str,**kwargs)
				else:
					fig = self.plot3d(uniques_col_str=uniques_col_str,**kwargs)
		except Exception as e:
			print('Error occured in quickplot: ',e)
		return fig

	# Automatic calibration of colorscale for colorplots
	def autoColorScale(self,data):
		#filter out NaNs or infinities, should any have crept in
		data = data[np.isfinite(data)]
		#print(data)
		m = 2.5 # n standard deviations to be considered as outliers
		datastd = data # make copy
		booleanmask = np.array([False]) # initialise boolean mask
		
		# Loop that recursively throws out outliers until remaining data fits in 2.5 times the std
		while False in booleanmask: 
		    if np.std(datastd) < 0.1*np.mean(np.abs(datastd)): # Check that data has a std larger than 10 of its mean, if not, break loop
		    	break
		    booleanmask = abs(datastd - np.mean(datastd)) < m * np.std(datastd) # make boolean mask based with outliers marked as False
		    if np.sum(booleanmask) < 0.8*len(data): #Loop to break the recursive std determination
		    	break
		    datastd = datastd[booleanmask] # apply mask to data
		
		#values, edges = np.histogram(datastd, 256) # bin for 256 colors in colorscale
		stretchfactor = .05 # stretching colorscale so that the data sits comfortably within its bounds
		cminlim = np.min(datastd)-((np.max(datastd)-np.min(datastd))*stretchfactor)
		cmaxlim = np.max(datastd)+((np.max(datastd)-np.min(datastd))*stretchfactor)
		return (cminlim,cmaxlim) # Return max and min lims for the colormap

	def plot3d(self,	massage_func=None, # For externally defined styles not defined in styles.py
						uniques_col_str=[], #
						drawCbar=True, # Switch colorbar
						cax_destination=None, # Specify destination colorbar axis, default 'None' makes new axis
						subplots_args={'top':0.96, 'bottom':0.17, 'left':0.14, 'right':0.85,'hspace':0.1,'wspace':0.1}, # Relative spacings and margins for subplots
						ax_destination=None,  # Specify destination data axis, default 'None' makes new axis
						n_index=None, # Index of higher dimensional measurement
						style=['normal'],  # Used styles for plotting
						xlims=None, # Set non-default x-axis limits
						ylims=None, # Set non-default x-axis limits
						clim=None, # Set non-default colorbar limits
						aspect='auto', # Control aspect ration of plot
						interpolation='nearest', # Interpolation in colorplot (data is unaffected)
						value_axis=-1, # Selects which subplot to plot (list), -1 plots all
						imshow=True, # Default uses imshow, else pcolormesh is used
						cbar_orientation='horizontal', # Orientataion of colorbar, either vertical or horizontal
						cbar_location ='normal', # 'inset' allows plotting colorbar inside the main fig
						filter_raw = True, # Filters out all subplots whith the word 'raw' in the name or label of the plot
						ccmap = None, #Allows loading of custom colormap
						supress_plot = False, #Suppression of all plotting functions, only processes data
						norm = 'nan', #Added for NaN value support
						axislabeltype = 'label', #Use 'label' or 'name' on axis labels
						quantiphy = True, #Use quantiphy to convert units and axis labels to manageable quantities
						**kwargs):
		#some housekeeping
		if not self.fig and not ax_destination:
			self.fig = plt.figure()
			self.fig.subplots_adjust(**subplots_args)
		if n_index != None:
			if len(n_index)==0:
				n_index = None
			
		#loading of colormap
		if ccmap:
			self.ccmap = loadCustomColormap()
		else:
			self.ccmap = copy.copy(mpl.cm.get_cmap("inferno"))		

		#make a list of uniques per column associated with column name
		value_keys_raw,value_units_raw,value_labels_raw = self.data.valuekeys_n
		coord_keys_raw,coord_units_raw,coord_labels_raw = self.data.coordkeys_n

		#Filtering raw value axes
		if filter_raw== True and self.isthumbnail:
			value_keys_filtered = []
			value_labels_filtered = []
			value_units_filtered = []
			for n,value_label_raw in enumerate(value_labels_raw):
				if value_label_raw.find('raw')==-1 and value_label_raw.find('Raw')== -1:
					value_labels_filtered.append(value_label_raw)
					value_keys_filtered.append(value_keys_raw[n])
					value_units_filtered.append(value_units_raw[n])
			value_keys = value_keys_filtered
			value_units = value_units_filtered
			value_labels = value_labels_filtered
		else:
			value_keys = value_keys_raw
			value_units = value_units_raw
			value_labels = value_labels_raw

		#make a list of uniques per column associated with column name
		uniques_by_column = dict(zip(self.data.coordkeys + self.data.valuekeys, self.data.dims))
		if len(uniques_by_column)>2:
			if not uniques_col_str:
				uniques_col_str = list(uniques_by_column)[0:-2]
			titlecube = 'Higher order measurement, '
		else:
			titlecube = ''
		# Collect keys, units and uniques
		coord_keys = [key for key in coord_keys_raw if key not in uniques_col_str ]
		coord_units = list(coord_units_raw[i] for i in [i for i, key in enumerate(coord_keys_raw) if key not in uniques_col_str])
		coord_labels = list(coord_labels_raw[i] for i in [i for i, key in enumerate(coord_keys_raw) if key not in uniques_col_str])
		unique_labels = list(coord_labels_raw[i] for i in [i for i, key in enumerate(coord_keys_raw) if key in uniques_col_str])

		value_axes = []	
		if type(value_axis) != list:
			value_axes = [value_axis]
		else:
			value_axes = value_axis	
		if value_axes[0] == -1:
			value_axes = list(range(len(value_keys)))

		if not self.isthumbnail:
			if n_index != None:
				n_index = set(n_index)
				n_subplots = len(n_index)
			else:
				n_subplots = len(list(self.data.make_filter_from_uniques_in_columns(uniques_col_str)))
			width = int(np.ceil(np.sqrt(len(value_axes*n_subplots))))
			height = int(np.ceil(len(value_axes*n_subplots)/width))
			
			gs = gridspec.GridSpec(height,width)
			for k in rcP:
				mpl.rcParams[k] = rcP[k]
			if ax_destination == None:
				self.fig.set_size_inches(width*_plot_width*np.sqrt(2)/np.sqrt(height), height*_plot_height*np.sqrt(2)/np.sqrt(height))
		else:
			if n_index != None:
				n_index = n_index[0]
				#n_subplots = 1
			else:
				n_subplots = len(list(self.data.make_filter_from_uniques_in_columns(uniques_col_str)))
			gs = gridspec.GridSpec(len(value_axes),1)
			for k in rcP:
				mpl.rcParams[k] = rcP_thumb[k]
			if ax_destination == None:
				self.fig.set_size_inches(_plot_width_thumb, (len(value_axes)-1)*2+_plot_height_thumb)
		
		cnt=0 #subplot counter

		#enumerate over the generated list of unique values specified in the uniques columns
		for j,ind in enumerate(self.data.make_filter_from_uniques_in_columns(uniques_col_str)):
			#print(j)
			#each value axis needs a plot

			for value_axis in value_axes:
				#plot only if number of the plot is indicated
				if not self.isthumbnail and n_index != None:
					if j not in n_index:
						continue
				if self.isthumbnail and j != 0:
					continue
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
					ystart = data_slice[coord_keys[-1]].iloc[-1]+ystep
					yend = missingpoints*ystep+ystart-ystep
					yarr = np.linspace(ystart,yend,missingpoints)
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
				ext = (x.min(),x.max(),y.min(),y.max())
				self.extent = ext
				
				#setting custom xlims and ylims, restricted within extent of data
				if xlims == None:
					_xlims = (ext[0],ext[1])
				else: #sets custum xlims only if they restrict the default xaxis
					xzip = list(zip(xlims,[ext[0],ext[1]]))
					xlimsl = 2*[None]
					xlimsl[0]=max(xzip[0])
					xlimsl[1]=min(xzip[1])
					_xlims=tuple(xlimsl)

				if ylims == None:
					_ylims = (ext[2],ext[3])
				else: #sets custom xyims only if they restrict the default yaxis
					yzip = list(zip(ylims,[ext[2],ext[3]]))
					ylimsl = 2*[None]
					ylimsl[0]=max(yzip[0])
					ylimsl[1]=min(yzip[1])
					_ylims=tuple(ylimsl)

				#Gridding and interpolating unevenly spaced data
				extx = abs(ext[1]-ext[0])
				xdx = np.diff(X, axis=0)
				xdxshape = xdx.shape
				if len(xdx) != 0:
					minxstep = np.nanmin(abs(xdx[xdx > 1e-19])) # removing rounding errors from diff
					minxsteps = int(round(extx/minxstep,0))+1
				else:
					minxsteps = extx
				
				exty = abs(ext[3]-ext[2])
				ydy = np.diff(Y, axis=1)#.astype(float)
				ydyshape = ydy.shape
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
				self.exportData.append(XX)
				try:
					m={
						'xu':xu,
						'yu':yu,
						'xlims':_xlims,
						'ylims':_ylims,
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
					data_quantity = value_labels[value_axis] 
				elif axislabeltype == 'name':
					xaxislabel = coord_keys[-2] 
					yaxislabel = coord_keys[-1]
					data_quantity = value_keys[value_axis] 
				else:
					raise Exception('Wrong axislabeltype argument, must be \'name\' or \'label\'')
				xaxisunit = coord_units[-2]
				yaxisunit = coord_units[-1]
				data_unit = value_units[value_axis]
				#wrap all needed arguments in a datastructure
				sbuffer = ''
				data_trans = [] #transcendental tracer :P For keeping track of logs and stuff
				w = styles.getPopulatedWrap(style)
				w2 = {
						'ext':ext,
						'XX': XX,
						'X': X,
						'Y': Y,
						'x': x,
						'y': y,
						'z': z,
						'data_quantity': data_quantity, 
						'data_unit': data_unit, 
						#'data_trans':data_trans, 
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
				Y = w['Y']
				X = w['X']
				data_trans_formatted = ''.join([''.join(s+'(') for s in w['data_trans']])
				#data_title = data_trans_formatted + w['data_quantity'] + ' (' + w['data_unit'] + ')'
				#if len(w['data_trans']) != 0:
				#	data_title = data_title + ')'
				self.stylebuffer = w['buffer'] 
				self.xlabel = w['xlabel']
				self.xunit = w['xunit']
				self.ylabel= w['ylabel']
				self.yunit = w['yunit']
				self.datalabel = w['data_quantity']
				self.dataunit = w['data_unit']
				self.XX_processed = XX
				self.Y_processed = Y
				self.X_processed = X
				self.extent_processed = ext
				xunit = self.xunit
				yunit = self.yunit
				ext = self.extent_processed
				dataunit = self.dataunit

				# Quantity conversion with the help of Quantiphy
				if quantiphy == True:
					if self.xunit != '' and self.xunit not in self.quantiphy_ignorelist:
						ind1 = np.nanargmax(np.abs(ext[0:2]))
						extqu1 = Quantity(ext[ind1], self.xunit).format().split(' ')
						convfactor1 = float(extqu1[0])/ext[ind1]
						self.quant_xunit = extqu1[1]
						xunit = self.quant_xunit
					else:
						convfactor1 = 1
					
					if self.yunit != '' and self.yunit not in self.quantiphy_ignorelist:
						ind2 = np.nanargmax(np.abs(ext[2:4]))+2
						extqu2 = Quantity(ext[ind2], self.yunit).format().split(' ')
						convfactor2 = float(extqu2[0])/ext[ind2]
						self.quant_yunit = extqu2[1]
						yunit = self.quant_yunit
					else:
						convfactor2 = 1	

					if self.dataunit != '' and self.dataunit not in self.quantiphy_ignorelist:
						dataind = np.divmod(np.nanargmax(np.abs(self.XX_processed)),  self.XX_processed.shape[1])
						dataqu = Quantity(self.XX_processed[dataind],self.dataunit).format().split(' ')
						dataconvfactor = float(dataqu[0])/self.XX_processed[dataind]
						self.quant_XX_processed = self.XX_processed * dataconvfactor
						self.quant_dataunit = dataqu[1]
						dataunit = self.quant_dataunit
						XX = self.quant_XX_processed

					self.quant_extent_processed = ext = (ext[0]*convfactor1,ext[1]*convfactor1,ext[2]*convfactor2,ext[3]*convfactor2)
					ext = self.quant_extent_processed

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
						colormap = (plt.get_cmap(self.ccmap))
						colormap.set_bad('grey',1.0)
						self.im = ax.imshow(np.rot90(XX), 
											extent=ext, 
											cmap=plt.get_cmap(self.ccmap), 
											aspect=aspect, 
											interpolation=interpolation, 
											norm=self.imshow_norm,
											clim=clim)
					else:
						xs = np.linspace(ext[0],ext[1],XX.shape[0])
						ys = np.linspace(ext[2],ext[3],XX.shape[1])
						xv,yv = np.meshgrid(xs,ys) 

						colormap = (plt.get_cmap(self.ccmap)) # Support for plotting NaN values
						colormap.set_bad('none',1.0)
						self.im = ax.pcolormesh(xv,yv,np.rot90(np.fliplr(XX)),
															   cmap=plt.get_cmap(self.ccmap), 
															   vmin=clim[0], 
															   vmax=clim[1])
				if not clim:
					try:
						self.im.set_clim(self.autoColorScale(XX.flatten()))
					except:
						pass
				if not (xlims==None) or not (ylims==None):
					ax.set_xlim(_xlims)
					ax.set_ylim(_ylims)
				#ax.locator_params(nbins=5, axis='y') #Added to hardcode number of x ticks.
				#ax.locator_params(nbins=7, axis='x')
				#Tclk = Quantity(10e-9, 'S')

				if 'flipaxes' in style:
					xaxislabelwithunit = self.ylabel +	' (' + yunit + ')'
					yaxislabelwithunit = self.xlabel +	' (' + xunit + ')'
				else:
					xaxislabelwithunit = self.xlabel +	' (' + xunit + ')'
					yaxislabelwithunit = self.ylabel +	' (' + yunit + ')'
				ax.set_xlabel(xaxislabelwithunit)
				ax.set_ylabel(yaxislabelwithunit)
				cbar_title = self.datalabel + ' (' + dataunit + ')'

				title = ''
				for h,i in enumerate(uniques_col_str):
					if quantiphy == True:
						titlequ = Quantity(getattr(data_slice,i).iloc[0],coord_units_raw[coord_keys_raw.index(i)]).format().split(' ')
						title = '\n'.join([title, '{:s}: {:s} {:s}'.format(unique_labels[h],titlequ[0], titlequ[1] )])
					else:
						title = '\n'.join([title, '{:s}: {:g} {:s}'.format(unique_labels[h],getattr(data_slice,i).iloc[0], coord_units_raw[coord_keys_raw.index(i)] )])

				if 'notitle' not in style:
					if not self.isthumbnail:
						ax.set_title(title, loc='left', pad=32, weight='bold')
					if self.isthumbnail:
						ax.set_title(titlecube + title, loc='left', pad=0, weight='bold',fontsize=10)						
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
							cax.set_aspect(0.07)
							cax.set_anchor('E')
						else:
							cax = divider.append_axes("right", size="2.5%", pad=0.05)
						pos = list(ax.get_position().bounds)
					if hasattr(self, 'im'):
						self.cbar = colorbar.create_colorbar(cax, self.im, orientation=cbar_orientation)
						cbar = self.cbar

						if cbar_orientation == 'horizontal': #Added some hardcode config for colorbar, more pretty out of the box
							tick_locator = ticker.MaxNLocator(nbins=3)
							cbar.locator = tick_locator
							cbar.set_label(cbar_title,labelpad=-17, x = -0.08, horizontalalignment='right')
							cbar.ax.xaxis.set_label_position('top')
							cbar.ax.xaxis.set_ticks_position('top')
						else:
							tick_locator = ticker.MaxNLocator(nbins=3)
							cbar.locator = tick_locator
							cbar.set_label(cbar_title)#,labelpad=-19, x=1.32)
						
						self.cbar = cbar
						cbar.update_ticks()
						if supress_plot == False:
							plt.tight_layout()
							plt.show()
				self.ax = ax
				cnt+=1 #counter for subplots
		
		
		if self.fig and (mpl.get_backend() in [qtaggregator , 'nbAgg']):
			self.toggleFiddle()
			self.toggleLinedraw()
			self.toggleLinecut()
			self.toggleWaterfall()
		return self.fig

	def plot2d(self,massage_func=None,
					uniques_col_str=[], #
					subplots_args={'top':0.96, 'bottom':0.17, 'left':0.14, 'right':0.85,'hspace':0.3,'wspace':0.6},
					ax_destination=None,
					n_index=None,
					style=['normal'],
					xlims=None,
					ylims=None,
					aspect='auto',
					value_axis = -1,
					fiddle=False,
					supress_plot = False,
					legend=False,
					filter_raw=True, #Do not show plots with 'Raw' string in label
					axislabeltype = 'label', #Use 'label' or 'name' on axis labels
					quantiphy = True, #Use quantiphy to convert units and axis labels to manageable quantities
					**kwargs):
					
		if not self.fig and not ax_destination:
			self.fig = plt.figure()
			self.fig.subplots_adjust(**subplots_args)
		if n_index != None:
			if len(n_index)==0:
				n_index = None            

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

		value_axes = []	
		if type(value_axis) != list:
			value_axes = [value_axis]
		else:
			value_axes = value_axis	
		if value_axes[0] == -1:
			value_axes = list(range(len(value_keys)))


		if not self.isthumbnail:
			width = int(np.ceil(np.sqrt(len(value_axes))))
			height = int(np.ceil(len(value_axes)/width))
			gs = gridspec.GridSpec(height,width)
			for k in rcP:
				mpl.rcParams[k] = rcP[k]
			if ax_destination == None:
				self.fig.set_size_inches(width*_plot_width, height*_plot_height)
		else:
			gs = gridspec.GridSpec(len(value_axes),1)
			for k in rcP:
				mpl.rcParams[k] = rcP_thumb[k]
				if ax_destination == None:
					self.fig.set_size_inches(_plot_width_thumb, (len(value_axes)-1)*2+_plot_height_thumb)

		for i,j in enumerate(self.data.make_filter_from_uniques_in_columns(uniques_col_str)):
		
			for k,value_axis in enumerate(value_axes):
				if n_index != None:
						if i not in n_index:
							continue
				data = self.data.sorted_data[j]
				#filter out the keys corresponding to unique value columns
				us=uniques_col_str
				coord_keys = [key for key in coord_keys if key not in uniques_col_str]
				#now find out if there are multiple value axes
				if not coord_keys:
					print('Warning: 2dplot found no coordinate axis, resetting keys. Repeat measurement of same coordinate suspected.')
					coord_keys,coord_units,coord_labels = self.data.coordkeys_n
				#value_keys, value_units = self.data.valuekeys
				x=data.loc[:,coord_keys[-1]]
				xx=data.loc[:,value_keys[value_axis]]

				if axislabeltype == 'label' and len(coord_labels_raw) == len(coord_keys_raw):
					xaxislabel = coord_labels[-1]
				else: #else defaulting to column name for axis labels
					xaxislabel = coord_keys[-1] 
				if axislabeltype == 'label' and len(value_labels) == len(value_keys):
					data_quantity = value_labels[value_axis]
				else: #else defaulting to column name for axis labels
					data_quantity = value_keys[value_axis]

				xaxisunit = coord_units[-1]
				data_unit = value_units[value_axis]

				npx = np.array(x)
				npxx = np.array(xx)
				self.XX = npxx
				self.X = npx
				self.x = x
				self.z = xx

				title =''

				# for i,z in enumerate(uniques_col_str):
				# 	pass
				# 	# this crashes sometimes. did not investiagte yet what the problem is. switched off in the meantime
				# 	title = '\n'.join([title, '{:s}: {:g}'.format(uniques_axis_designations[i],data[z].iloc[0])])
				wrap = styles.getPopulatedWrap(style)
				wrap['XX'] = npxx
				wrap['X']  = npx
				wrap['Y'] = np.nan #not existing
				wrap['xlabel'] = xaxislabel
				wrap['xunit'] = xaxisunit
				wrap['data_quantity'] = data_quantity
				wrap['data_unit'] = data_unit
				wrap['massage_func'] = massage_func
				styles.processStyle(style,wrap)

				self.stylebuffer = wrap['buffer'] 
				self.xlabel = wrap['xlabel']
				self.xunit = wrap['xunit']
				self.datalabel= wrap['data_quantity']
				self.dataunit = wrap['data_unit']
				self.XX_processed = wrap['XX']
				self.X_processed = wrap['X']
				xunit = self.xunit
				dataunit = self.dataunit
				XX = self.XX_processed
				X = self.X_processed

				# Quantity conversion with the help of Quantiphy
				if quantiphy == True:
					if self.xunit != '' and self.xunit not in self.quantiphy_ignorelist:
						extqu1 = Quantity(np.nanmax(np.abs(self.X_processed)), self.xunit).format().split(' ')
						convfactor1 = float(extqu1[0])/np.nanmax(np.abs(self.X_processed))
						self.quant_xunit = extqu1[1]
						xunit = self.quant_xunit
						X = self.X*convfactor1
					
					if self.dataunit != '' and self.dataunit not in self.quantiphy_ignorelist:
						dataqu = Quantity(np.nanmax(np.abs(self.XX_processed)),self.dataunit).format().split(' ')
						dataconvfactor = float(dataqu[0])/np.nanmax(np.abs(self.XX_processed))
						self.quant_XX_processed = self.XX_processed * dataconvfactor
						self.quant_dataunit = dataqu[1]
						dataunit = self.quant_dataunit
						XX = self.quant_XX_processed

				if supress_plot == False:
					#setting custom xlims and ylims, restricted within extent of data
					if xlims == None:
						_xlims = (X.min(),X.max())
					else: #sets custum xlims only if they restrict the default xaxis
						xzip = list(zip(xlims,[X.min(),X.max()]))
						xlimsl = 2*[None]
						xlimsl[0]=max(xzip[0])
						xlimsl[1]=min(xzip[1])
						_xlims=tuple(xlimsl)

					if ylims == None:
						_ylims = (XX.min(),XX.max())
					else: #sets custum xyims always (since this is the data axis)
						_ylims=tuple(ylims)

					xaxislabelwithunit = wrap['xlabel'] + ' (' + xunit + ')'
					yaxislabelwithunit = wrap['data_quantity'] + ' (' + dataunit + ')'
					if ax_destination:
						ax = ax_destination
					else:
						ax = plt.subplot(gs[k])
					ax.plot(X,XX,'o-', fillstyle='none', markersize=2,label=title,**kwargs)
					#self.cutAx.plot(xx,z,'o-',fillstyle='none',markersize=2)
					if legend:
						plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
							   ncol=2, mode="expand", borderaxespad=0.)
					if ax:
						ax.set_xlabel(xaxislabelwithunit)
						ax.set_ylabel(yaxislabelwithunit)
					if not (xlims==None) or not (ylims==None):
						ax.set_xlim(_xlims)
						ax.set_ylim(_ylims)
					plt.tight_layout()
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

		#attach to the relevant figure to make sure the object does not go out of scope
		self.fig.linedraw = self.linedraw

	def toggleLinecut(self):
		self.linecut=Linecut(self.fig,self)
		self.fig.cutbutton = toggleButton('cut', self.linecut.connect)
		topwidget = self.fig.canvas.window()
		toolbar = topwidget.children()[1]
		action = toolbar.addWidget(self.fig.cutbutton)

		#attach to the relevant figure to make sure the object does not go out of scope
		self.fig.linecut = self.linecut

	def toggleWaterfall(self):
		self.waterfall=Waterfall(self.fig,self)
		self.fig.waterfallbutton = toggleButton('waterfall', self.waterfall.connect)
		topwidget = self.fig.canvas.window()
		toolbar = topwidget.children()[1]
		action = toolbar.addWidget(self.fig.waterfallbutton)

		#attach to the relevant figure to make sure the object does not go out of scope
		self.fig.waterfall = self.waterfall

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