# tessierplot
## Installation

Go into the project folder and type:
```
python setup.py install
```
If you're feeling frisky
```
python setup.py develop
```
installs in develop mode and the project folder is where the module
lives. Any editing done will immediately carry over.


## Usage

### tessier view
Make your life easier by making thumbnails of all measurement files
and plot them in a nice grid.
```python
import imp
from tessierplot import view
imp.reload(view)

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
p.plot3d(style=['mov_avg(n=1,m=2)','didv'])
```
applies a moving average filtering, and a subsequent derivative. Other
filters at this time are:

```python
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

### the colorbar
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


