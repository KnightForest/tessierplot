
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
    dir,basename =  os.path.split(file)
    #print(datedir, basename)    
    #print(os.path.split(os.path.dirname(file))[1])
    cachepath = os.path.normpath(os.path.join(os.getcwd(),'thumbnails', datedir + os.path.split(os.path.dirname(file))[1] + '_'+basename+ '_thumb.png'))
    #print(cachepath)
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
    if file_Extension == '.dat' or file_Extension == '.csv':
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
    def __init__(self, rootdir='./', filemask='.*\.dat(?:\.gz)?$|.*\.csv(?:\.gz)?$',filterstring='',override=False,headercheck=None,style=[],showfilenames=False):
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
                if len(p.data) > 5: ##just make sure really unfinished measurements are thrown out
                    is2d = p.is2d()
                    if not style:
                        if is2d:
                            guessStyle = ['normal']
                        else :
                            guessStyle = ['normal']
                            #guessStyle = p.guessStyle()
                    else:
                        guessStyle = []

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
            basenames = []
            #print(filenames)
            #in the current directory find all files matching the filemask
            for filename in filenames:
                fullpath = os.path.join(root,filename)
                res = reg.findall(filename)
                #print(res)
                if res:
                    dir,basename =  os.path.split(fullpath)
                    basenames.append(basename) #collect basenames in folder (no ext)
                    #print(basenames)
                    if (basename in s for s in basenames): #if basename does not exist, append
                        matches.append(fullpath)
            #found at least one file that matches the filemask
            #print(matches)
            for fullpath in matches: 
                #fullpath = matches[match]

                # check if filterstring can be found in the path
                isinfilterstring = filterstring.lower() in fullpath.lower()

                dir,basename =  os.path.split(fullpath)

                #extract the directory which is the date of measurement
                datedir = os.path.basename(os.path.normpath(dir+'/../'))
                measfiledir = os.path.basename(os.path.normpath(dir))

                #print(measfiledir)
                if os.name == 'nt': # avoid problems with very long path names in windows
                   dir = win32api.GetShortPathName(dir)
                   fullpath = dir + '/' + basename
                
                measname,ext1 = os.path.splitext(basename)
                #dirty, if filename ends e.g. in gz, also chops off the second extension
                measname,ext2 = os.path.splitext(measname)
                ext = ext2+ext1
                #fulldir = os.path.dirname(fullpath)
                #lastdir = os.path.basename(fulldir)
                if len(measname) < len(measfiledir):
                    measname = measfiledir
                #check if filterstring can be found in the set file (e.g. 'dac4: 1337.0')
                if not isinfilterstring:
                    setfilepath = data.filetype.getsetfilepath(fullpath)
                    if setfilepath: # only check for filterstring if set file exists
                        isinfilterstring = filterstring in open(setfilepath).read()

                if isinfilterstring:    #liable for improvement
                    if self._showfilenames:
                        print(fullpath)
                    df = data.Data.load_header_only(fullpath)
                    #print(df._header)
                    #print(df.coordkeys)
                    value_keys = [i['name'] for n,i in enumerate(df._header) if ('column' in df._header[n] and i['type']=='value')]
                    coord_keys = [i['name'] for n,i in enumerate(df._header) if ('column' in df._header[n] and i['type']=='coordinate')]

                    # Ensures only the full file is loaded for higher order measurements (unavoidable)
                    if len(df.coordkeys) > 2:
                        df = data.Data.from_file(fullpath)
                        higherdim_coords = [df.coordkeys[0:-2],df.dims[0:-2],[],[]]
                        higherdim_coords[2] = [[] for i in range(len(higherdim_coords[0]))]
                        for i,coords in enumerate(higherdim_coords[0]):
                            higherdim_coords[2][i] = getattr(df.sorted_data,coords).unique()
                        higherdim_coords[3] = df.coordkeys_n[1]
                    else:
                        higherdim_coords = None
                    if headercheck is None or coord_keys[-2] == headercheck: # # Filtering thumbnails when headercheck matches a coordinate axis
                        thumbpath = self.makethumbnail(fullpath,**kwargs)

                        if thumbpath:
                            thumbpath_html = thumbpath.replace('#','%23') # html does not like number signs in file paths
                            if 'comment' in df._header[-1]:
                                comment = (df._header[-1]['comment'])
                            else:
                                comment = 'n.a.'
                            self._allthumbs.append({'datapath':fullpath,
                                                 'thumbpath':thumbpath_html,
                                                 'datedir':datedir, 
                                                 'measname':measname,
                                                 'comment':comment,
                                                 'value_keys':value_keys,
                                                 'higherdim_coords':higherdim_coords})
                            images += 1
        return self._allthumbs

    def _ipython_display_(self):
        
        display_html(HTML(self.genhtml(refresh=False,style=self._style)))
        
    def genhtml(self,refresh=False,**kwargs):
        if self._override:
            refresh = True
        self.walk(filemask=self._filemask,
                  filterstring=self._filterstring,
                  headercheck=self._headercheck,
                  override=refresh,
                  **kwargs) #Change override to True if you want forced refresh of thumbs
        
        #unobfuscate the file relative to the working directory
        #since files are served from ipyhton notebook from ./files/
        all_relative = [{ 
                            'thumbpath':'./files/'+os.path.relpath(k['thumbpath'],start=os.getcwd()),
                            'datapath': k['datapath'], 
                            'datedir': k['datedir'], 
                            'measname': k['measname'],
                            'comment': k['comment'],
                            'value_keys': k['value_keys'],
                            'higherdim_coords': k['higherdim_coords'] } for k in self._allthumbs]
        out=u"""
        
        <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
        <meta http-equiv="Pragma" content="no-cache"> 
        <meta http-equiv="Expires" content="0">
        
        <div id='outer'>
    
    {% set ncolumns = 4 %}
    {% set vars = {'lastdate': '', 'columncount': 1, 'higherorder': 0,} %}
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
            <div id='{{ item.datapath }}' class='col'>  {# mousehover filename and measurement comment #}
                <div class='name'> {{ item.measname + '\n' + 'Comment: ' + item.comment }} </div>
                <div class='thumb'>
                        <img src="{{ item.thumbpath }}?{{ nowstring }}"/> 
                </div>
                <div class='controls'>
                    <button id='{{ item.datapath }}' onClick='toclipboard(this.id)'>Filename to clipboard</button>
                    <br/>
                    <button id='{{ item.datapath }}' onClick='refresh(this.id)'>Refresh</button>
                    <button id='{{ item.datapath }}' onClick='plotwithStyle(this.id)' class='plotStyleSelect'>Plot with</button>
                    <form name='{{ item.datapath }}'>
                    Style:
                    <select name="selector"> 
                        <option value="{{"\\'\\'"|e}}">normal</option>
                        <option value="{{"\\'int\\'"|e}}">int</option>
                        <option value="{{"\\'log\\'"|e}}">log</option>
                        <option value="{{"\\'logdb\\'"|e}}">logdb</option>
                        <option value="{{"\\'mov_avg(n=12)\\'"|e}}">movavg</option>
                        <option value="{{"\\'savgol(samples=7,order=3,difforder=0)\\'"|e}}">Savitzky-Golay_filter</option>
                        <option value="{{"\\'mov_avg(n=3)\\',\\'diff(condquant=False)\\',\\'mov_avg(n=3)\\'"|e}}">diff_movavg</option>
                        <option value="{{"\\'mov_avg(n=3)\\',\\'diff(condquant=False)\\',\\'mov_avg(n=3)\\',\\'log\\'"|e}}">diff_movavg_log</option>
                        <option value="{{"\\'mov_avg(n=3)\\',\\'diff(condquant=True)\\',\\'mov_avg(n=3)\\'"|e}}">diff_movavg_condquant</option>
                        <option value="{{"\\'mov_avg(n=12)\\',\\'diff(condquant=False)\\',\\'mov_avg(n=12)\\'"|e}}">diff_movavg_smooth</option>
                        <option value="{{"\\'savgol(condquant=False,samples=7,order=3)\\'"|e}}">Savitzky-Golay_diff</option>
                        <option value="{{"\\'savgol(condquant=True,samples=7,order=3,difforder=1)\\'"|e}}">Savitzky-Golay_diff_conquant</option>
                        <option value="{{"\\'savgol(condquant=False,samples=7,order=3,difforder=1)\\',\\'log\\'"|e}} ">Savitzky-Golay_diff,log</option>
                        <option value="{{"\\'movingmeansubtract(window=1)\\'"|e}} ">movmeansubtract</option>
                        <option value="{{"\\'movingmediansubtract(window=1)\\'"|e}} ">movmediansubtract</option>
                        <option value="{{"\\'meansubtract\\',\\'mov_avg(n=3)\\',\\'ivreverser\\',\\'mov_avg(n=3)\\',\\'diff\\'"|e}}">ivreverser,diff</option>
                        <option value="{{"\\'int\\',\\'meansubtract\\',\\'ivreverser\\',\\'diff\\'"|e}}">int,ivreverser,diff</option>
                        <option value="{{"\\'int\\',\\'meansubtract\\',\\'ivreverser\\',\\'diff\\'"|e}}">int,mov_avg,ivreverser,diff</option>
                        <option value="{{"\\'unwrap\\'"|e}}">unwrap</option>
                        <option value="{{"\\'unwrap\\',\\'diff\\',\\'mov_avg(n=6)\\'"|e}}">unwrap,diff,mov_avg</option>
                        <option value="{{"\\'meansubtract\\',\\'deinterlace0\\'"|e}} ">deinterlace0</option>
                        <option value="{{"\\'meansubtract\\',\\'deinterlace1\\'"|e}} ">deinterlace1</option>
                        <option value="{{"\\'meansubtract\\',\\'deinterlace0\\',\\'mov_avg\\',\\'diff\\'"|e}} ">deinterlace0,diff</option>
                        <option value="{{"\\'meansubtract\\',\\'deinterlace1\\',\\'mov_avg\\',\\'diff\\'"|e}} ">deinterlace1,diff</option>
                    </select>
                    </br>
                    Measurement axis:
                    <select name="value_axis">
                        <option value="{{'-1'}}">All</option>
                        {% for key in item.value_keys %}
                            <option value="{{ loop.index0 }}">{{ key }}</option>
                        {% endfor %}
                    </select>
                    </br>
                    <input type="checkbox" name="stylechecker" value="{{"\\'flipaxes\\',"|e}} ">Flip axes
                    </br>
                    {% if (item.higherdim_coords != None) %}
                        {% for key in item.higherdim_coords[0] %}
                            {{ key }} 
                            <select name="n_index{{ loop.index0 }}">
                                <option value="{{''}}">All</option>
                                {% set prevloop = loop.index0 %}
                                {% for key in range(item.higherdim_coords[1][loop.index0]) %}
                                    <option value="{{ loop.index0 }}">{{ item.higherdim_coords[2][prevloop][key] }}</option>
                                {% endfor %}
                            </select>
                            {{ item.higherdim_coords[3][loop.index0] }}
                            </br>
                            <input type="hidden" name="higherorderdims{{ loop.index0 }}" value="{{ item.higherdim_coords[1][loop.index0] }}">
                        {% endfor %}
                        {% if vars.update({'higherorder': item.higherdim_coords[0]|length}) %} {% endif %}
                    {% else %} 
                        {% if vars.update({'higherorder': 0}) %} {% endif %}
                    {% endif %}
                    <input type="hidden" name="higherorder" value="{{ vars.higherorder }}">
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
                var style = getStyle(id);
                id = id.replace(/\\\\/g,"\\\\\\\\");
                //window.alert(style);
                exec ='from tessierplot import view;  a=view.tessierView();a.makethumbnail(\"' + id + '\",override=True,style=%s)';
                exec=exec.printf(style);
                pycommand(exec,refresh_callback,id); 
            }
            function refresh_callback(id) {
                var x = document.querySelectorAll('div[id=\\"'+id+'\\"]')[0];
                var img = x.getElementsByTagName('img')[0];
                img.src = img.src.split('?')[0] + '?' + new Date().getTime();
            }
            function getStyle(id) {
                id = id.replace(/\\\\/g,"\\\\\\\\"); // changing single backslash to double backslash in filenamepath
                var x = document.querySelectorAll('form[name=\\"'+id+'\\"]')
                form = x[0]; //should be only one form
                selector = form.selector;
                var stylevalues = selector.options[selector.selectedIndex].value
                //window.alert(stylevalues);
                if(form.stylechecker.checked) {
                    stylevalues = form.stylechecker.value + stylevalues
                }
                var style = "["{{" + stylevalues + "|e}}"]";
                return style
            }
            function getValue_axis(id) {
                id = id.replace(/\\\\/g,"\\\\\\\\");
                var x = document.querySelectorAll('form[name=\\"'+id+'\\"]')
                form = x[0]; //should be only one form
                value_axis = form.value_axis;
                var v_ax = value_axis.options[value_axis.selectedIndex].value
                //window.alert(v_ax)
                return v_ax
            }
            function getHO_index(id) {
                id = id.replace(/\\\\/g,"\\\\\\\\");
                var x = document.querySelectorAll('form[name=\\"'+id+'\\"]');
                form = x[0]; //should be only one form
                //window.alert( form.higherorder.value );
                if (form.higherorder.value > 0) {
                    var n_ind_temp = parseInt(0)
                    var coorddim = parseInt(1)
                    //window.alert('fHO ' + form.higherorder.value)
                    for (var i = form.higherorder.value - 1; i >= 0; i--) {
                        var str = 'n_index' + i;
                        var str2 = 'higherorderdims' + i;
                        //window.alert(str + ': ' + form[str].value);
                        //window.alert(str2 + ': ' + form[str2].value);
                        var coordindex = parseInt(form[str].value);
                        //window.alert('cindex:' + coordindex + ', cdimprev' + coorddim);
                        //n_ind_temp = n_ind_temp * coorddim + coordindex
                        n_ind_temp = n_ind_temp + coordindex * coorddim
                        coorddim = parseInt(form[str2].value);
                        //window.alert('n_ind_temp' + ': ' + n_ind_temp);
                    }
                    if (isNaN(n_ind_temp)) {
                        n_ind=''
                    } else {
                        n_ind=n_ind_temp
                    }
                } else {
                    var n_ind = ''
                }
                //window.alert('n_indfinal'+n_ind)
                
                return n_ind
            }
            function plotwithStyle(id) {
                var style = getStyle(id);
                var v_ax = getValue_axis(id);
                var n_ind = getHO_index(id);
                //window.alert(n_ind);
                //var value_axis = form.value_axis;
                plot(id,style,v_ax,n_ind);
            }
            function plot(id,style,v_ax,n_ind){
                //window.alert(style)
                //window.alert(v_ax)
                //window.alert(n_ind)
                dir = id.split('/');
                
                exec = 'filename \= \"' + id + '\"; {{ plotcommand }}';
                //window.alert(exec)
                exec = exec.printf(style,v_ax,n_ind)
                //window.alert(exec)
                pycommand(exec);
            }
            String.prototype.printf = function (style,v_ax,n_ind) {
                var useArguments = false;
                var _arguments = arguments;
                var i = -1;
                if (typeof _arguments[0] == "string") {
                useArguments = true;
                }
                if (style instanceof Array || useArguments) {
                var that = this.replace(/\%c/g,n_ind)
                that = that.replace(/\%a/g,v_ax)
                return that.replace(/\%s/g,
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
                  return style[i];
                });
                }
                else {
                return that.replace(/{([^{}]*)}/g,
                function (a, b) {
                  var r = style[b];
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
                    white-space: pre-line;
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
            width:75%;
            height:auto;
            }
        </style>
        """
        temp = jj.Template(out)

        plotcommand = """\\nimport matplotlib.pyplot as plt\\nimport imp\\nif not plt.get_fignums():\\n from tessierplot import plot as ts\\n imp.reload(ts)\\np = ts.plotR(filename)\\np.quickplot(style=%s,value_axis=[%a],n_index=[%c])\\n"""
        
        import datetime
        d=datetime.datetime.utcnow()
        nowstring = d.strftime('%Y%m%d%H%M%S')
        return temp.render(items=all_relative,plotcommand=plotcommand,nowstring=nowstring)