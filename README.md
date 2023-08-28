# tessierplot
## Installation

Go into the project folder and type:
```
pip install {path to package}
```
For an editable install (recommended for easy repository updates):
```
pip install -e {path to package}
```
Now, the project folder is where the module lives. 
Any editing done will immediately carry over.


## Usage

### tessier view
Make your life easier by making thumbnails of all measurement files
and plot them in a nice grid.
```python
import importlib		
from tessierplot import view
importlib.reload(view)

view.tessierView(rootdir='/where/my/naturepublicationmeasurements/are',filterstring='',override=False, showfilenames=True)
```

As can be seen tessierView takes 4 potential arguments:
- The rootdir is where view begins recursively looking for files matching the filterstring. 
- The filterstring is expected to be a regular expression.
- override determines if datafile preview files are replotted, even if they already exist. The default value is False.
- showfilenames determines if the file name of every plotted data file is displayes. The default value is False

### plotR
This is the main plotting object, it takes a measurement file as
argument, after which a specific command can be given to plot either 2d, 3d , or some
other type of plot.

```python
from tessierplot import plot

p = plot.plotR('mymeasurementfilelocation.dat.gz')
p.quickplot() #automagically figures out if it's a 2d or 3d plot and plots accordingly
p.plot2d() #plot in 2d
p.plot3d() #plot in..hey 3d.
p.starplot() #starplot measurement files only, only plot datapoints for each separate axis
```

### uniques_col_str ###

By supplying a ```uniques_col_str``` parameter to a plot command the
way the data is segmented can be altered.

E.g. the data consists of 4 data columns.
x y z r

with x,y,z coordinates and r a value that needs plotting. For a 3d
plot, values y,z are most likely the coordinates with r being the
corresponding value needing plotting. Sometimes x is varied slowly
taking only a limited amount (n) of unique values. Thus, it would be
logical to plot only n 3d plots with the value of x being indicated
per plot.

You can manually supply which columns contain these 'unique' coordinates. In
this particular example one could supply ```p.plot3d(uniques_col_str=['x'])``` or, if y is also pretty sparse ```p.plot2d(uniques_col_str=['x','y'])``` .

#### styles
plotR support several default ''styles'' that can be applied to the
data to make your life a bit easier.

e.g.

```python
p.plot3d(style=['mov_avg(n=1,m=2)','diff'])
```
applies a moving average filtering, and a subsequent derivative. Other
filters at this time are:

```python
STYLE_SPECS = {
	'abs': {'param_order': []},
	'changeaxis': {'xfactor': 1,'xoffset':0,'xlabel':None,'xunit':None, 'yfactor':1,'yoffset':0,'ylabel':None,'yunit':None, 'datafactor':1,'dataoffset':0,'dataquantity':None,'dataunit':None, 'param_order': ['xfactor','xoffset','xlabel','xunit', 'yfactor','yoffset','ylabel','yunit', 'datafactor','dataoffset','dataquantity','dataunit']},
	'crosscorr': {'peakmin':None,'peakmax':None,'toFirstColumn':True,'param_order': ['peakmin','peakmax','toFirstColumn']},
	'dbmtovolt': {'rfamp': False, 'attenuation': 0, 'gridresolutionfactor': 2, 'param_order': ['rfamp','attenuation','gridresolutionfactor']},
	'deint_cross': {'param_order': []},
	'deinterlace': {'param_order': []},
	'deinterlace0': {'param_order': []},
	'deinterlace1': {'param_order': []},
	'deleteouterdatapoints': {'n':0,'param_order': ['n']},
	'diff': {'condquant': False, 'axis': 0, 'gradient': True, 'order': 1, 'param_order': ['condquant','axis','gradient','order']},
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
	'ivreverser':{'gridresolutionfactor': 10, 'twodim': False, 'interpmethod': 'cubic', 'param_order': ['gridresolutionfactor','twodim','interpmethod']},
	'linecut': {'linecutvalue': 1,'axis': None, 'quantiphy' : True, 'param_order': ['linecutvalue','axis', 'quantiphy']},
	'log': {'param_order': []},
	'logdb': {'param_order': []},
	'massage': {'param_order': []},
	'meansubtract': {'param_order': []},
	'minsubtract': {'param_order': []},
	'normalise': {'axis': 'x', 'index': 0, 'param_order': ['axis','index']},
	'mov_avg': {'m': 1, 'n': 3, 'win': None, 'param_order': ['m', 'n', 'win']},
	'movingmeansubtract': {'window': 2,'param_order': ['window']},
    'movingmediansubtract': {'window': 1,'param_order': ['window']},
	'normal': {'param_order': []},
	'offsetslopesubtract': {'slope': 0, 'offset': 0, 'param_order': ['slope', 'offset']},
	'resistance': {'linecutvalue': 0, 'dolinearfit': False, 'fitregion': 1, 'param_order': ['linecutvalue','dolinearfit','fitregion']},
	'rshunt': {'r':1e-10,'gridresolutionfactor': 2, 'param_order': ['r','gridresolutionfactor']},
	'savgol': {'condquant': False, 'axis': 0, 'difforder':1, 'samples': 7, 'order': 3, 'param_order': ['condquant','axis','difforder','samples','order']},
	'sgtwodidv': {'samples': 21, 'order': 3, 'param_order': ['samples', 'order']},
	'shapiro': {'rffreq': 2.15e9, 'nsteps': 1, 'millivolts': False, 'param_order': ['rffreq','nsteps','millivolts']},
	'unwrap': {'param_order': []},
	'vbiascorrector':{'voffset': 0,'seriesr': 0, 'gridresolutionfactor': 2, 'param_order': ['voffset','seriesr','gridresolutionfactor']},
}

```
#### The massage style
you can also custom make a style without modifying the tessierplot module.

Supplying the plot command with a
```massage_func=special_style```. Will cause ```special_style``` to be called
and you can process the data inline.

```python
def special_style(wrapper):
	#do some fancy manipulation of your data
	wrapper['X'] #xdata in 2d, empty for 3d
	wrapper['XX'] #ydata/zdata, depending on 2d or 3d

	return wrapper
p.quickplot(style=['mov_avg', 'massage', 'didv'], massage_func=special_style)
```

### the colorbar [functionality currently removed]
The colorbar supports modifying the colormap nonlinearly by clicking in it. This
will divide the colormap in n+1 segments, where n is the number of
marks. Each mark can be dragged.

e.g. for 2 marks, the modifying points will be at 1/3 and 2/3 of the
colormap. Dragging a mark will effectively drag these points to an
arbitrary new point, scaling the colormap in the process.

### fiddle, linedraw, and linecut
By clicking one of the three buttons in the 3d plot window it's
possible to modify and extract more info from a plot.

Fiddle changes the colormap range by clicking and dragging in the figure once
fiddle has been enabled. Look to the colorbar for the effect. This
feature is useful to bring out features that are maybe otherwise
hidden.

With linedraw enabled, you can draw a line, and observe its length and
slope.

Linecut gives a linecut of the data. For a horizontal linecut, hold
the alt-key.


