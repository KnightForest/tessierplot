from matplotlib.colorbar import ColorbarBase
from matplotlib import cbook
import matplotlib.colors as colors
import numpy as np

class SuperColorbar(ColorbarBase):

    '''
    Implement ColorbarBase a bit like mpl.colors.ColorBar so that when a mappable changed also the colorbar g
    gets updates along with it
    Limited in scope, since it doesnt take care if mappable is e.g. a contour.ContourSet
    '''
    def __init__(self,ax,mappable,**kwargs):
        #self.ticklocation='top'
        self._normticks = []
        self._normtickhandles = []
        self._dragtick = None #current tick being dragged
        if ax != None:
            self.fig = ax.get_figure()

        self.mappable = mappable
        kwargs['cmap'] = cmap = mappable.cmap
        kwargs['norm'] = norm = mappable.norm
        self.callbacksSM = cbook.CallbackRegistry()
        self._update_dict = {'array': False}

        
        self.cidpress = self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig.canvas.mpl_connect(
                'button_release_event', self.on_release)
        self.cidmotion = self.fig.canvas.mpl_connect(
                'motion_notify_event', self.on_motion)

        super(SuperColorbar,self).__init__(ax,**kwargs)

    def on_release(self, event):
        #all is awell
        self.press = None
        self._dragtick = None
    def on_press(self, event):
        contains, attrd = self.ax.contains(event)
        if not contains: return
        if(event.xdata==None): return
        self.press = self.get_pos(event)
        if self.press is None:
            return
        a = event.inaxes.images
    
        #see if press is on previous normalization tick (within 10% of scale)
        margin = .1
        #if so drag will be on said tick
        for tick in self._normticks:
           if self.press - margin < tick < self.press + margin:
                self._dragtick = tick
                break
        else:
            self._dragtick = self.press
            #else create a new tick
            self._normticks.append(self.press)
        self.update_normticks()
        self.update_mappable()
    def on_motion(self, event):
        #change the normalization tick position
        
        if self._dragtick:
            pos = self.get_pos(event)
            if pos != None:
                ind = self._normticks.index(self._dragtick)
                self._normticks[ind] = pos
                self._dragtick = pos
                self.update_normticks()
                self.update_mappable()
    def get_pos(self,event):
        #determine either horizontal or vertical mode
        if self.orientation == 'horizontal':
            press = event.xdata
        else:
            press = event.ydata
        return press
    def update_normticks(self):
        colors = [[1,0,0] for t in self._normticks]
        #renormalize ticks on colorbar to values
        locs = self.norm.inverse(self._normticks)
        self.add_lines(locs, colors, 5., erase=True)
        self.fig.canvas.draw()
    def update_mappable(self):
        self.norm.set_points(self.norm.inverse(self._normticks))
        self.draw_all()
        self.mappable.autoscale()
        self.fig.canvas.draw()
    # def set_clim(self, vmin=None, vmax=None):
    #     """
    #     Set the norm limits for image scaling.
    #     Parameters
    #     ----------
    #     vmin, vmax : float
    #          The limits.
    #          The limits may also be passed as a tuple (*vmin*, *vmax*) as a
    #          single positional argument.
    #          .. ACCEPTS: (vmin: float, vmax: float)
    #     """
    #     if vmax is None:
    #         try:
    #             vmin, vmax = vmin
    #         except (TypeError, ValueError):
    #             pass
    #     if vmin != None:
    #         self.norm.vmin = colors._sanitize_extrema(vmin)
    #     if vmax != None:
    #         self.norm.vmax = colors._sanitize_extrema(vmax)
    #     self.changed()
    # def set_cmap(self, cmap):
    #     """
    #     Set the colormap for luminance data.
    #     Parameters
    #     ----------
    #     cmap : `.Colormap` or str or None
    #     """
    #     in_init = self.cmap is None
    #     #cmap = get_cmap(cmap)
    #     self.cmap = cmap
    #     #if not in_init:
    #     self.changed()  # Things are not set up properly yet.
    # def changed(self):
    #     """
    #     Call this whenever the mappable is changed to notify all the
    #     callbackSM listeners to the 'changed' signal.
    #     """
    #     self.callbacksSM.process('changed', self)
    #     for key in self._update_dict:
    #         self._update_dict[key] = True
    #     self.stale = True
    def on_mappable_changed(self, mappable):
        cm = mappable.get_cmap()
        cl = mappable.get_clim()
        self.set_cmap(cm)
        self.set_clim(cl)
        self.draw_all()
        if self.orientation == 'horizontal':
            self.ax.xaxis.set_label_position('top')
            self.ax.xaxis.set_ticks_position('top')

class MultiPointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, points=None, clip=False):
        self.points = points
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def set_points(self,points):
        self.points = points
    def __call__(self, value, clip=None):
        # Ignoring masked values and all kinds of edge cases
        if self.points is None:
            points = [(self.vmax - self.vmin)/2. + self.vmin]
        else:
            points = self.points

        points = np.array(points)
        points.sort()
        x = np.hstack((self.vmin, points,self.vmax))
        y = np.linspace(0, 1, len(points)+2)

        return np.ma.masked_array(np.interp(value, x, y))

if __name__ == "__main__":
    test()

def create_colorbar(ax,mappable,*args,**kwargs):
    cb=SuperColorbar(ax,mappable,*args,**kwargs)
    #create callback to notify supercolorbar that the mappable has changed
    #cid = mappable.callbacksSM.connect('changed',cb.on_mappable_changed)
    #mappable.colorbar = cb
    #mappable.colorbar_cid= cid
    return cb
    
def test():
    import matplotlib.pyplot as plt
    from matplotlib import cm
    fig = plt.figure()
    ax1=fig.add_axes([0,0,.8,1])
    #add normalizeable dude
    mappable = ax1.imshow([[5,2],[0,1]],cmap=cm.viridis,interpolation='nearest',norm=MultiPointNormalize())

    ax2=fig.add_axes([0.8,0,1,1])
    create_colorbar(ax2,mappable)
    plt.show()  

    


    
