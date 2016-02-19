#tessierMagic, where the real % happens
#tessierView for couch potato convenience...

import tessierPlot as ts
import jinja2 as jj
import matplotlib.pyplot as plt
import pylab
import os
from itertools import chain
import numpy as np
import re
from IPython.display import VimeoVideo
from IPython.display import display, HTML, display_html


reload(ts)

_plot_width = 4. # in inch (ffing inches eh)
_plot_height = 3. # in inch

_fontsize_plot_title = 10
_fontsize_axis_labels = 10
_fontsize_axis_tick_labels = 10

plotstyle = 'normal'

rcP = {  'figure.figsize': (_plot_width, _plot_height), #(width in inch, height in inch)
        'axes.labelsize':  _fontsize_axis_labels,
        'xtick.labelsize': _fontsize_axis_tick_labels,
        'ytick.labelsize': _fontsize_axis_tick_labels,
        'legend.fontsize': 5.
        }
        
rcP_old = pylab.rcParams.copy()

# pylab.rcParams['legend.linewidth'] = 10
def getthumbcachepath(file):
    oneupdir = os.path.abspath(os.path.join(os.path.dirname(file),os.pardir))
    datedir = os.path.split(oneupdir)[1] #directory name should be datedir, if not 
    if re.match('[0-9]{8}',datedir):
        preid= datedir
    else:
        preid = ''
    
    #relative to project/working directory
    cachepath = os.path.normpath(os.path.join(os.getcwd(),'thumbnails', preid + '_'+os.path.splitext(os.path.split(file)[1])[0] + '_thumb.png'))
    return cachepath
    
def getthumbdatapath(file):
    thumbdatapath = os.path.splitext(file)[0] + '_thumb.png'
    return thumbdatapath


def overridethumbnail(file, fig):
##Todo, copy thumbs on the fly from their working directories, thus negating having to recheck..the..data..file?
    #create a thumbnail and store it in the same directory and in the thumbnails dir for local file serving, override options for if file already exists
    thumbfile = getthumbcachepath(file)
    thumbfile_datadir =  getthumbdatapath(file)
    try:
        pylab.rcParams.update(rcP)
        fig.savefig(thumbfile,bbox_inches='tight' )
        fig.savefig(thumbfile_datadir,bbox_inches='tight' )
        plt.close(fig)
    except Exception,e:
        thumbfile = None #if fail no thumbfile was created
        print 'Error {:s} for file {:s}'.format(e,file)
        pass
        
    #put back the old settings
    pylab.rcParams = rcP_old

    return thumbfile


class tessierView(object):
    def __init__(self, rootdir='./', filemask='.*\.dat(?:\.gz)?$',filterstring=''):
        self._root = rootdir
        self._filemask = filemask
        self._filterstring = filterstring
        self._allthumbs = []
    def on(self):   
        print 'You are now watching through the glasses of ideology'
        display(VimeoVideo('106036638'))
          
    def getsetfilepath(self,file):
        file_Path, file_Extension = os.path.splitext(file)
        if   file_Extension ==  '.gz':
            file_Path = os.path.splitext(file_Path)[0]
        elif file_Extension != '.dat':
            print 'Wrong file extension'

        return (file_Path + '.set')
    def makethumbnail(self, file,override=False,style=[]):
        #create a thumbnail and store it in the same directory and in the thumbnails dir for local file serving, override options for if file already exists
        thumbfile = getthumbcachepath(file)
        #print file
        thumbfile_datadir =  getthumbdatapath(file)
        try:
            if os.path.exists(thumbfile):
                thumbnailStale = os.path.getmtime(thumbfile) < os.path.getmtime(file)   #check modified date with file modified date, if file is newer than thumbfile refresh the thumbnail
            if ((not os.path.exists(thumbfile)) or override or thumbnailStale):
                #now make thumbnail because it doesnt exist or if u need to refresh
                pylab.rcParams.update(rcP)
                p = ts.plotR(file)
                if len(p.data) > 20: ##just make sure really unfinished measurements are thrown out
                    is2d = p.is2d()
                    if is2d:
                        guessStyle = ['normal']
                    else :
                        guessStyle = p.guessStyle()
                    p.quickplot(style=guessStyle + style)

                    p.fig.savefig(thumbfile,bbox_inches='tight' )
                    p.fig.savefig(thumbfile_datadir,bbox_inches='tight' )
                    plt.close(p.fig)
                    
                else:
                    thumbfile = None
        except Exception,e:
            thumbfile = None #if fail no thumbfile was created
            print 'Error {:s} for file {:s}'.format(e,file)
            pass
            
        #put back the old settings
        pylab.rcParams = rcP_old

        return thumbfile


    def walklevel(self,some_dir, level=1):
        some_dir = some_dir.rstrip(os.path.sep)
        assert os.path.isdir(some_dir)
        num_sep = some_dir.count(os.path.sep)
        for root, dirs, files in os.walk(some_dir):
            yield root, dirs, files
            num_sep_this = root.count(os.path.sep)
            if num_sep + level <= num_sep_this:
                del dirs[:]
            
    def walk(self,filemask,filterstring,**kwargs):
        paths = (self._root,)
        images = 0
        self.allthumbs = []
        reg = re.compile(self._filemask) #get only files determined by filemask
        for root,dirnames,filenames in chain.from_iterable(os.walk(path) for path in paths):
            matches = []
            #in the current directory find all files matching the filemask
            for filename in filenames:
                fullpath = os.path.join(root,filename)
                #print fullpath
                res = reg.findall(filename)
                if res:
                    matches.append(fullpath)
            #found at least one file that matches the filemask
            if matches: 
                fullpath = matches[0]                
                dir,basename =  os.path.split(fullpath)
                
                measname,ext1 = os.path.splitext(basename)
                #dirty, if filename ends e.g. in gz, also chops off the second extension
                measname,ext2 = os.path.splitext(measname)
                ext = ext2+ext1
                
                #extract the directory which is the date of measurement
                datedir = os.path.basename(os.path.normpath(dir+'/../'))
                

                if filterstring in open(self.getsetfilepath(fullpath)).read():   #liable for improvement
                #check for certain parameters with filterstring in the set file: e.g. 'dac4: 1337.0'
                    thumbpath = self.makethumbnail(fullpath,**kwargs)
                    if thumbpath:
                        self._allthumbs.append({'datapath':fullpath,
                                             'thumbpath':thumbpath,
                                             'datedir':datedir, 
                                             'measname':measname})
                        images += 1

        return self._allthumbs
    def _ipython_display_(self):
        display_html(HTML(self.genhtml(refresh=False)))
        from IPython.display import Javascript,display_javascript
        a = Javascript('jump(\'#endofpage\')')
        display_javascript(a)
    
    def htmlout(self,refresh=False):
        display(HTML(self.genhtml(refresh=refresh)))
        
    def genhtml(self,refresh=False):
        self.walk(self._filemask,'dac',override=refresh) #Change override to True if you want forced refresh of thumbs
        #unobfuscate the file relative to the working directory
        #since files are served from ipyhton notebook from ./files/
        all_relative = [{ 
                            'thumbpath':'./files/'+os.path.relpath(k['thumbpath'],start=os.getcwd()),
                            'datapath':k['datapath'], 
                            'datedir':k['datedir'], 
                            'measname':k['measname'] } for k in self._allthumbs]

        out=u"""

        
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache"> 
        <meta http-equiv="Expires" content="0">
        
        <div id='outer'>
    {% set columncount = 1 %}
    {% set ncolumns = 3 %}
    {% set lastdate = '' %}
    {% for item in items %}
    
        {% set isnewdate = (lastdate != item.datedir) %}
        {% if isnewdate %}
            {% set columncount = 1 %}
            {% if loop.index != 1 %}
                </div> {# close previous row, but make sure no outer div is closed #}
            {% endif %}
            <div class='datesep'> {{item.datedir}} </div>                
        {% endif %}
        
        {% if (columncount % ncolumns == 1) %}
            <div class='row'>
        {% endif %}

            <div class='col'>

                <div class='name'> {{ item.measname }} </div>

                <div class='thumb'>
                        {% if loop.index == items|length %}
                        <a name="#endofpage"></a>
                        {% endif %}
                        <img src="{{ item.thumbpath }}"/> 
                </div>

                <div class='controls'>
                    <button id='{{ item.datapath }}' onClick='toclipboard(this.id)'>Filename to clipboard</button>
                    <br/>
                    <button id='{{ item.datapath }}' onClick='refresh(this.id)'>Refresh</button>
                    <br/>
                    <button id='{{ item.datapath }}' onClick='plotwithStyle(this.id)' class='plotStyleSelect'>Plot with</button>
                    <form name='{{ item.datapath }}'>
                    <select name="selector">
                        <option value="{{"[\\'\\']"|e}}">normal</option>
                        <option value="{{"[\\'abs\\']"|e}}">abs</option>
                        <option value="{{"[\\'log\\']"|e}}">log</option>
                        <option value="{{"[\\'savgol\\',\\'log\\']"|e}}">savgol,log</option>
                        <option value="{{"[\\'sgdidv\\']"|e}}">sgdidv</option>
                        <option value="{{" [\\'sgdidv\\',\\'log\\']"|e}} ">sgdidv,log</option>
                        <option value="{{" [\\'mov_avg(m=1,n=15)\\',\\'didv\\',\\'mov_avg(m=1,n=15)\\',\\'abs\\',\\'log\\' ]"|e}} ">Ultrasmooth didv</option>
                        <option value="{{" [\\'mov_avg\\',\\'didv\\',\\'abs\\']"|e}} ">mov_avg,didv,abs</option>
                        <option value="{{" [\\'mov_avg\\',\\'didv\\',\\'abs\\',\\'log\\']"|e}} ">mov_avg,didv,abs,log</option>
                        <option value="{{" [\\'deinterlace0\\',\\'log\\']"|e}} ">deinterlace</option>
                        <option value="{{" [\\'crosscorr\\']"|e}} ">Crosscorr</option>
                    </select>

                    </form>            
                </div>
            </div>
        {% if (columncount % ncolumns == 0) %}
            </div>
        {% endif %}

        {% set lastdate = item.datedir %}
        {% set columncount = columncount + 1 %}
    {% endfor %}    
    </div>
    
    
        <script type="text/Javascript">
            function handle_output(out){
                    
            }
            function pycommand(exec){
                exec = exec.replace(/\\\\/g,"\\\\\\\\");
                var kernel = IPython.notebook.kernel;
                var callbacks = { 'iopub' : {'output' : handle_output}};
                var msg_id = kernel.execute(exec, callbacks, {silent:false});
            }
            function jump(h){
                var url = location.href;               //Save down the URL without hash.
                location.href = "#"+h;                 //Go to the target element.
                history.replaceState(null,null,url);   //Don't like hashes. Changing it back.
            }
            function tovar(id) {
                exec =' file \= \"' + id + '\"';
                pycommand(exec);
            }
            function toclipboard(id) {
                exec ='import pyperclip; pyperclip.copy(\"' + id + '\");pyperclip.paste()';
                pycommand(exec);
            }
            function refresh(id) {
                exec =' import tessierView as tv;  a=tv.tessierView();a.makethumbnail(\"' + id + '\",override=True)';
                pycommand(exec);
            }
            function getStyle(id) {
                id = id.replace(/\\\\/g,"\\\\\\\\");
                var x = document.querySelectorAll('form[name=\\"'+id+'\\"]')
                form = x[0]; //should be only one form
                selector = form.selector;
                var style = selector.options[selector.selectedIndex].value;
                return style
            }
            function plotwithStyle(id) {
                var style = getStyle(id);
                plot(id,style);
            }
            
            function plot(id,x,y){
                dir = id.split('/');
                
                exec = 'file \= \"' + id + '\"; {{ plotcommand }}';
                exec = exec.printf(x)

                pycommand(exec);
            }
            String.prototype.printf = function (obj) {
                var useArguments = false;
                var _arguments = arguments;
                var i = -1;
                if (typeof _arguments[0] == "string") {
                useArguments = true;
                }
                if (obj instanceof Array || useArguments) {
                return this.replace(/\%s/g,
                function (a, b) {
                  i++;
                  if (useArguments) {
                    if (typeof _arguments[i] == 'string') {
                      return _arguments[i];
                    }
                    else {
                      throw new Error("Arguments element is an invalid type");
                    }
                  }
                  return obj[i];
                });
                }
                else {
                return this.replace(/{([^{}]*)}/g,
                function (a, b) {
                  var r = obj[b];
                  return typeof r === 'string' || typeof r === 'number' ? r : a;
                });
                }
            };

        </script>
        
        <style type="text/css">
         .container { width:95% !important; } 
        @media (min-width: 30em) {
            .row {  width: auto; 
                    display: table; 
                    table-layout: fixed; 
                    padding-top: 1em; 
                    }
            .col { display: table-cell;  
                    padding-right: 1em;                
                    position:relative;
                    }
        }
        .col .name { z-index:5;
                    font-size: 10pt; 
                    color:black; 
                    position:absolute; 
                    top: 2px; 
                    width:80% ;
                    border: solid black 1px;
                    background-color: white;
                    word-break:break-all; 
                    opacity:0;
                    -moz-transition: all .2s;
                    -webkit-transition: all .2s;
                    transition: all .2s;
                    }
        .col:hover .name {
                    opacity:1;
                }            
        
        .datesep { width:100%; 
                   height: auto; 
                   border-bottom: 2pt solid black; 
                   font-size:14pt;
                   font-weight:bold;
                   font-style: italic;
                   padding-top: 2em;}
        #outer {}
        img{
            width:100%;
            height:auto;
            }
        </style>
        """
        temp = jj.Template(out)

        plotcommand = """import tessierPlot as ts; reload(ts); p = ts.plotR(file); p.quickplot(style= %s) """      
        return temp.render(items=all_relative,plotcommand=plotcommand)