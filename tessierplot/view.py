
import matplotlib.pyplot as plt
from . import plot as ts
from . import data
import jinja2 as jj
import os
from itertools import chain
import numpy as np
import re
from IPython.display import VimeoVideo
from IPython.display import display, HTML, display_html
import gzip
import shutil
#import time

if os.name == 'nt':
    import win32api

import imp
imp.reload(ts)

plotstyle = 'normal'

#pylab.rcParams['legend.linewidth'] = 10
def getthumbcachepath(file):
    oneupdir = os.path.abspath(os.path.join(os.path.dirname(file),os.pardir))
    datedir = os.path.split(oneupdir)[1] #directory name should be datedir, if not 
        
    #relative to project/working directory
    cachepath = os.path.normpath(os.path.join(os.getcwd(),'thumbnails', datedir + '_'+os.path.split(os.path.dirname(file))[1] + '_thumb.png'))

    return cachepath
    
def getthumbdatapath(file):
    f,ext = os.path.splitext(file)

    # make sure we have stripped of all wrapping file extesions due to e.g. zipping
    while not ext=='':
        f,ext = os.path.splitext(f)

    thumbdatapath = f + '_thumb.png'
    return thumbdatapath

def gzipit(self): #compresses all .dat files to .gz, deletes .dat files.
    file_Path, file_Extension = os.path.splitext(self)
    if file_Extension == '.dat':
        print(self)
        gzfile = self + '.gz'
        if os.path.isfile(gzfile)==False:
            with open(self, 'rb') as f_in, gzip.open(self+'.gz', 'wb') as f_out:
                print('Do the GZIP, do the GZIP')
                shutil.copyfileobj(f_in, f_out)
        os.unlink(self)
    elif file_Extension == '.gz':
        datfile = self[:-3]
        if os.path.isfile(datfile)==True:
            #print datfile
            os.unlink(datfile)

class tessierView(object):
    def __init__(self, rootdir='./', filemask='.*\.dat(?:\.gz)?$',filterstring='',override=False,headercheck=None,style=[],showfilenames=False):
        self._root = rootdir
        self._filemask = filemask
        self._filterstring = filterstring
        self._allthumbs = []
        self._headercheck = headercheck
        self._style = style
        self._override = override
        self._showfilenames = showfilenames
        
        #check for and create thumbnail dir
        thumbnaildir = os.path.dirname(getthumbcachepath('./'))
        if not os.path.exists(thumbnaildir):
            try:
                os.mkdir(thumbnaildir)
            except:
                raise Exception('Couldn\'t create thumbnail directory')
        
    def on(self):   
        print('You are now watching through the glasses of ideology')
        display(VimeoVideo('106036638'))
    
    def makethumbnail(self, filename,override=False,style=[]):
        #create a thumbnail and store it in the same directory and in the thumbnails dir for local file serving, override options for if file already exists
        gzipall = 0 #finds all uncompressed files and compresses them. Also deletes all *.dat files. Only use if short on diskspace.
        if gzipall == 1:
            gzipit(file)

        thumbfile = getthumbcachepath(filename)
        
        thumbfile_datadir =  getthumbdatapath(filename)
        
        try:
            if os.path.exists(thumbfile):
                thumbnailStale = os.path.getmtime(thumbfile) < os.path.getmtime(filename)   #check modified date with file modified date, if file is newer than thumbfile refresh the thumbnail
            if ((not os.path.exists(thumbfile)) or override or thumbnailStale):
                #now make thumbnail because it doesnt exist or if u need to refresh
                p = ts.plotR(filename,isthumbnail=True,thumbs = [thumbfile,thumbfile_datadir])

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
                while True: #destroy the plotR object
                    try: 
                        p
                    except:
                        break
                    else:                        
                        del p
    
        except Exception as e:
            thumbfile = None #if fail no thumbfile was created
            print('Error {:s} for file {:s}'.format(str(e),filename))
            pass
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
            
    def walk(self, filemask, filterstring, headercheck=None,**kwargs):
        paths = (self._root,)
        images = 0
        self._allthumbs = []
        reg = re.compile(self._filemask) #get only files determined by filemask
        
        for root,dirnames,filenames in chain.from_iterable(os.walk(path) for path in paths):
            dirnames.sort(reverse=True)
            matches = []
            #in the current directory find all files matching the filemask
            for filename in filenames:
                fullpath = os.path.join(root,filename)
                res = reg.findall(filename)
                if res:
                    matches.append(fullpath)
            #found at least one file that matches the filemask
            if matches: 
                fullpath = matches[0]

                # check if filterstring can be found in the path
                isinfilterstring = filterstring.lower() in fullpath.lower()

                dir,basename =  os.path.split(fullpath)

                #extract the directory which is the date of measurement
                datedir = os.path.basename(os.path.normpath(dir+'/../'))
                
                if os.name == 'nt': # avoid problems with very long path names in windows
                    dir = win32api.GetShortPathName(dir)
                    fullpath = dir + '/' + basename
                
                measname,ext1 = os.path.splitext(basename)
                #dirty, if filename ends e.g. in gz, also chops off the second extension
                measname,ext2 = os.path.splitext(measname)
                ext = ext2+ext1

                #check if filterstring can be found in the set file (e.g. 'dac4: 1337.0')
                if not isinfilterstring:
                    setfilepath = data.filetype.getsetfilepath(fullpath)
                    if setfilepath: # only check for filterstring if set file exists
                        isinfilterstring = filterstring in open(setfilepath).read()

                if isinfilterstring:    #liable for improvement
                    if self._showfilenames:
                        print(fullpath)
                    df = data.Data.load_header_only(fullpath)
                    if headercheck is None or df.coordkeys[-2] == headercheck:
                        thumbpath = self.makethumbnail(fullpath,**kwargs)

                        if thumbpath:
                            thumbpath_html = thumbpath.replace('#','%23') # html does not like number signs in file paths
                            self._allthumbs.append({'datapath':fullpath,
                                                 'thumbpath':thumbpath_html,
                                                 'datedir':datedir, 
                                                 'measname':measname})
                            images += 1
        return self._allthumbs

    def _ipython_display_(self):
        
        display_html(HTML(self.genhtml(refresh=False,style=self._style)))
        
    def genhtml(self,refresh=False,**kwargs):
        if self._override:
            refresh = True
        self.walk(filemask=self._filemask,filterstring=self._filterstring,headercheck=self._headercheck,override=refresh,**kwargs) #Change override to True if you want forced refresh of thumbs
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
    
    {% set ncolumns = 4 %}
    {% set vars = {'lastdate': '', 'columncount': 1} %}

    {% for item in items %}
    
        {% set isnewdate = (vars.lastdate != item.datedir) %}
        {% if isnewdate %}
            {% if vars.update({'columncount': 1}) %} {% endif %}
            {% if loop.index != 1 %}
                </div> {# close previous row, but make sure no outer div is closed #}
            {% endif %}
            <div class='datesep'> {{item.datedir}} </div>                
        {% endif %}
        
        {% if (vars.columncount % ncolumns == 1) %}
            <div class='row'>
        {% endif %}

            <div id='{{ item.datapath }}' class='col'>

                <div class='name'> {{ item.measname }} </div>

                <div class='thumb'>
                        <img src="{{ item.thumbpath }}?{{ nowstring }}"/> 
                </div>

                <div class='controls'>
                    <button id='{{ item.datapath }}' onClick='toclipboard(this.id)'>Filename to clipboard</button>
                    <br/>
                    <button id='{{ item.datapath }}' onClick='refresh(this.id)'>Refresh</button>
                    <br/>
                    <button id='{{ item.datapath }}' onClick='plotwithStyle(this.id)' class='plotStyleSelect'>Plot with</button>
                    <form name='{{ item.datapath }}'>
                    <select name="selector">
                        <option value="{{"\\'fixlabels\\',\\'\\'"|e}}">normal</option>
                        <option value="{{"\\'fixlabels\\',\\'abs\\'"|e}}">abs</option>
                        <option value="{{"\\'fixlabels\\',\\'log\\'"|e}}">log</option>
                        <option value="{{"\\'fixlabels\\',\\'savgol\\',\\'log\\'"|e}}">savgol,log</option>
                        <option value="{{"\\'fixlabels\\',\\'sgdidv\\'"|e}}">sgdidv</option>
                        <option value="{{"\\'fixlabels\\',\\'sgdidv\\',\\'log\\'"|e}} ">sgdidv,log</option>
                        <option value="{{"\\'fixlabels\\',\\'sgtwodidv\\'"|e}}">sg2nddidv</option>
                        <option value="{{"\\'fixlabels\\',\\'mov_avg(m=2,n=5)\\',\\'didv\\',\\'mov_avg(m=2,n=5)\\',\\'abs\\',\\'log\\' "|e}} ">Ultrasmooth didv</option>
                        <option value="{{"\\'fixlabels\\',\\'mov_avg\\',\\'didv\\',\\'abs\\'"|e}} ">mov_avg,didv,abs</option>
                        <option value="{{"\\'fixlabels\\',\\'mov_avg(m=1,n=25)\\',\\'didv\\',\\'didv\\',\\'abs\\'"|e}} ">mov_avg,2nddidv,abs</option>
                        <option value="{{"\\'fixlabels\\',\\'mov_avg\\',\\'didv\\',\\'abs\\',\\'log\\'"|e}} ">mov_avg,didv,abs,log</option>
                        <option value="{{"\\'fixlabels\\',\\'meansubtract\\',\\'ivreverser\\',\\'mov_avg(n=3)\\',\\'didv\\'"|e}}">ivreverser,didv</option>
                        <option value="{{"\\'fixlabels\\',\\'meansubtract\\',\\'deinterlace0\\'"|e}} ">deinterlace0</option>
                        <option value="{{"\\'fixlabels\\',\\'meansubtract\\',\\'deinterlace1\\'"|e}} ">deinterlace1</option>
                        <option value="{{"\\'fixlabels\\',\\'meansubtract\\',\\'deinterlace0\\',\\'mov_avg\\',\\'didv\\'"|e}} ">deinterlace0,didv</option>
                        <option value="{{"\\'fixlabels\\',\\'meansubtract\\',\\'deinterlace1\\',\\'mov_avg\\',\\'didv\\'"|e}} ">deinterlace1,didv</option>
                        <option value="{{"\\'fixlabels\\',\\'crosscorr\\'"|e}} ">Crosscorr</option>
                    </select>
                    <input type="checkbox" name="stylechecker" value="{{"\\'flipaxes\\',"|e}} ">Flip axes
                    </form>            
                </div>
            </div>
        {% if (vars.columncount % ncolumns == 0) %}
            </div>
        {% endif %}

        {% if vars.update({'columncount': vars.columncount+1}) %} {% endif %}
        {% if vars.update({'lastdate': item.datedir}) %} {% endif %}
    {% endfor %}    
    </div>
    
    
        <script type="text/Javascript">
            var py_callbacks =[];
            function handle_output(out){
                // done executing python output now filter on which event
                if (out.msg_type == "execute_result") {
                    for (var key in py_callbacks) {
                        key = py_callbacks[key]
                        if (out.parent_header.msg_id == key.msg_id) {
                            //call the callback with arguments
                            key.cb(key.cb_args);
                            //and remove from the list
                            py_callbacks.pop(key); 
                        }
                    }
                }
            }
            function pycommand(exec,cb,cb_args){
                exec = exec.replace(/\\\\/g,"\\\\\\\\");
                var kernel = IPython.notebook.kernel;
                var callbacks = { 'iopub' : {'output' : handle_output}};
                var msg_id = kernel.execute(exec, callbacks, {silent:false});
                
                if (cb != undefined) { 
                    py_callbacks.push({msg_id: msg_id, cb: cb, cb_args: cb_args});
                }
            }
            function jump(h){
                var url = location.href;               //Save down the URL without hash.
                location.href = "#"+h;                 //Go to the target element.
                history.replaceState(null,null,url);   //Don't like hashes. Changing it back.
            }
            function tovar(id) {
                exec =' filename \= \"' + id + '\"';
                pycommand(exec);
            }
            function toclipboard(id) {
                exec ='import pyperclip; pyperclip.copy(\"' + id + '\");pyperclip.paste()';
                pycommand(exec);
            }
            function refresh(id) {
                id = id.replace(/\\\\/g,"\\\\\\\\");

                exec ='from tessierplot import view;  a=view.tessierView();a.makethumbnail(\"' + id + '\",override=True)';
                pycommand(exec,refresh_callback,id);
                
            }
            function refresh_callback(id) {
                var x = document.querySelectorAll('div[id=\\"'+id+'\\"]')[0];
                var img = x.getElementsByTagName('img')[0];
                img.src = img.src.split('?')[0] + '?' + new Date().getTime();
            }
            function getStyle(id) {
                id = id.replace(/\\\\/g,"\\\\\\\\");
                var x = document.querySelectorAll('form[name=\\"'+id+'\\"]')
                form = x[0]; //should be only one form
                selector = form.selector;
                var stylevalues = selector.options[selector.selectedIndex].value
                if(form.stylechecker.checked) {
                    stylevalues = form.stylechecker.value + stylevalues
                }
                var style = "["{{" + stylevalues + "|e}}"]";
                return style
            }
            function plotwithStyle(id) {
                var style = getStyle(id);
                plot(id,style);
            }
            
            function plot(id,x,y){
                dir = id.split('/');
                
                exec = 'filename \= \"' + id + '\"; {{ plotcommand }}';
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

        plotcommand = """\\nimport matplotlib.pyplot as plt\\nimport imp\\nif not plt.get_fignums():\\n from tessierplot import plot as ts\\n imp.reload(ts)\\np = ts.plotR(filename)\\np.quickplot(style= %s)\\n"""
        
        import datetime
        d=datetime.datetime.utcnow()
        nowstring = d.strftime('%Y%m%d%H%M%S')
        return temp.render(items=all_relative,plotcommand=plotcommand,nowstring=nowstring)
