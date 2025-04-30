import pandas
import os
import re
import numpy as np
from six.moves import xrange
import json
from collections import Counter
from itertools import count

class parser(object):
    def __init__(self):
        self._header = None        
        self._data = None
    def parse(self):
        return self._data
    def parseheader(self):
        pass

class dat_parser(parser):
    def __init__(self,filename=None,filebuffer=None):
        self._file = filename
        self._filebuffer = filebuffer

        super(dat_parser,self).__init__()

    def parse(self):
        filebuffer = self._filebuffer
        if filebuffer == None:
            f = open(self._file, mode='rb')
            self._filebuffer = f
        else:
            f = filebuffer
            self._filebuffer = filebuffer
        
        self._header,self._headerlength = self.parseheader()
        # Checks for duplicate names and adds numbers to them
        #print(self._header)
        names = [i['name'] for n,i in enumerate(self._header) if 'column' in self._header[n]]
        c = Counter(names)
        iters = {k: count(1) for k, v in c.items() if v > 1}
        names = [x+'_'+str(next(iters[x])) if x in iters else x for x in names]
        print(self._headerlength)
        self._data = pandas.read_csv(f,
                                 sep='\t+|,\\s+',
                                 comment='#',
                                 skiprows=self._headerlength,
                                 engine ='python',
                                 header=None,
                                 names=names)
        return super(dat_parser,self).parse()
        f.close()
        print(self._data)

    def parse_header(self):
        return None
    
    def is_valid(self):
        pass

class qtm_parser(dat_parser):
    def __init__(self,filename=None,filebuffer=None):
        super(qtm_parser,self).__init__(filename=filename,filebuffer=filebuffer)

    def parse(self):
        return super(qtm_parser,self).parse()

    def is_valid(self):
        pass

    def parseheader(self):
        filebuffer = self._filebuffer
        firstline = (filebuffer.readline().decode('utf-8', 'ignore')).rstrip()
        
        if not firstline: # for emtpy data files
            return None,-1
        headerlength = 3
        secondline = (filebuffer.readline().decode('utf-8', 'ignore')).rstrip()
        thirdline = (filebuffer.readline().decode('utf-8', 'ignore')).rstrip()
        #thirdline = (filebuffer.readline().decode('utf-8', 'ignore'))#.rstrip()
        setgetlist = list(firstline.split('|')[1])
        #print(secondline)
        columnnames = thirdline.split(', ')
        c = Counter(columnnames)
        iters = {k: count(1) for k, v in c.items() if v > 1}
        columnnames = [x+'_'+str(next(iters[x])) if x in iters else x for x in columnnames]
        headervalues=[]
        #units = []
        headerlength=0
        for i,val in enumerate(setgetlist):
            if val == 's':
                line = [i,columnnames[i],'coordinate','',columnnames[i]]
                line_x = line_x = zip(['column','name','type', 'unit', 'label'],line)
                headervalues.append(line_x)
            if val == 'g':
                line = [i,columnnames[i],'value','',columnnames[i]]
                line_x = line_x = zip(['column','name','type', 'unit', 'label'],line)
                headervalues.append(line_x)
        headervalues = [dict(x) for x in headervalues]

        return headervalues,headerlength 

class qcodes_parser(dat_parser):
    def __init__(self,filename=None,filebuffer=None):
        super(qcodes_parser,self).__init__(filename=filename,filebuffer=filebuffer)

    def parse(self):
        return super(qcodes_parser,self).parse()

    def is_valid(self):
        pass

    def parseheader(self):
        #read in the .json file
        json_f = [f for f in os.listdir(os.path.dirname(self._file)) if f.endswith('.json')]
        if len(json_f) > 1:
            raise ValueError('Too many snapshots in folder.')
        if len(json_f) < 1:
            raise ValueError('Cannot locate snapshot.')
        json_file = ''.join((os.path.dirname(self._file),'\\',json_f[0]))
        json_filebuffer = open(json_file)
        json_s=json_filebuffer.read()

        json_data = json.loads(json_s)
        
        #For the old loop methon of measuring: 
        if 'arrays' in json_data:

            #read the column names from the .dat file
            filebuffer = self._filebuffer
            firstline = (filebuffer.readline().decode('utf-8')).rstrip()
            secondline = filebuffer.readline().decode('utf-8')     
            raw2 = r'\".*?\"'
            reggy2 = re.compile(raw2)
            columnname2 = reggy2.findall(secondline)
            columnname2 = [i.replace('\"','') for i in columnname2]
            columnname = re.split(r'\t+', firstline)
            columnname[0] = columnname[0][2::]
                             
            #look for the part where the data file meta info is stored
            json_data = json.loads(json_s)
            headerdict = json_data['arrays']
            headervalues=[]
            units = []
            headerlength=0

            for i,val in enumerate(headerdict):
                if headerdict[val]['is_setpoint']:                
                    headerdictval = [i,headerdict[val]['array_id']][1]
                    headerdictunit = [i,headerdict[val]['unit']][1]
                    headerdictlabel = [i,headerdict[val]['label']][1]
                    line=[i,headerdictval,'coordinate', headerdictunit, headerdictlabel]
                    line_x = zip(['column','name','type', 'unit', 'label'],line)
                    headervalues.append(line_x)

                else:
                    headerdictval = [i,headerdict[val]['array_id']][1]
                    headerdictunit = [i,headerdict[val]['unit']][1]
                    headerdictlabel = [i,headerdict[val]['label']][1]
                    line=[i,headerdictval,'value', headerdictunit, headerdictlabel]
                    line_x = zip(['column','name','type', 'unit', 'label'],line)
                    headervalues.append(line_x)

            headervalues = [dict(x) for x in headervalues]
            # sort according to the column order in the dat file
            header=[]
            for i, col in enumerate(columnname):
                for j, h in enumerate(headervalues):
                        if col == h['name']:
                            h['name'] = columnname2[i] #Names in columns are more correct than in JSON.
                            header.append(h)
                            break
        
        #With json file from qcodes database format:
        if 'interdependencies' in json_data:
            #read the column names from the .dat file
            filebuffer = self._filebuffer
            filebuffer.seek(0)
            headerlines = []
            for i, linebuffer in enumerate(filebuffer):
                line = linebuffer.decode('utf-8', 'ignore').rstrip()
                if line[0] != '#': #find the skiprows accounting for the first linebreak in the header
                    headerlength = i-1
                    break
                headerlines.append(line)
                if i > 1e9: #Claudius wants infinite endlines in his comments, such metadata
                    break
            firstline = headerlines[0]
            if headerlength > 4:
                commentlist = headerlines[1:-3]
                secondline = ''
                for i, elem in enumerate(commentlist):
                    secondline += elem[2:] + '\n'
            else:
                secondline = headerlines[1]
            thirdline = headerlines[-2]
            fourthline = headerlines[-1]

            names = thirdline[2:].split('\t')
            
            headervalues=[]
            units = []
            headerlength=0

            headerdict = json_data['interdependencies']['paramspecs']
            for i,val in enumerate(headerdict):             
                if not headerdict[i]['depends_on']:                
                    headerdictval = [i,headerdict[i]['name']][1]
                    headerdictlabel = [i,headerdict[i]['label']][1]
                    headerdictunit = [i,headerdict[i]['unit']][1]
                    line=[i,headerdictval,headerdictlabel,'coordinate',headerdictunit]
                    line_x = zip(['column','name','label','type','unit'],line)
                    headervalues.append(line_x)
                else:
                    headerdictval = [i,headerdict[i]['name']][1]
                    headerdictlabel = [i,headerdict[i]['label']][1]
                    headerdictunit = [i,headerdict[i]['unit']][1]
                    line=[i,headerdictval,headerdictlabel,'value',headerdictunit]
                    line_x = zip(['column','name','label','type','unit'],line)
                    headervalues.append(line_x)
            headervalues = [dict(x) for x in headervalues]
            # print('headervalues',headervalues)
            # sort according to the column order in the dat file
            header=[]
            for i, col in enumerate(names):
                for j, h in enumerate(headervalues):
                        if col == h['name']:
                            header.append(h)
                            break
            titleline = firstline.split(',')
            comment = secondline
            if comment == 'Comment:':
                comment = 'n.a.'
            else:
                comment = comment.replace('Comment: ', '')
            #print('titline',titleline)
            headertitledict = {
            'measname'   : titleline[0],
            'experiment' : titleline[1],
            'samplename' : titleline[2],
            'nvals'      : titleline[3],
            'comment'    : comment
            }
            header.append(headertitledict)
        #print(json.dumps(header, sort_keys=True, indent=4))
        json_filebuffer.close()
        return header,headerlength #git c,headertitledict

class qtlab_parser(dat_parser):
    def __init__(self,filename=None,filebuffer=None):
        super(qtlab_parser,self).__init__(filename=filename,filebuffer=filebuffer)

    def parse(self):
        return super(qtlab_parser,self).parse()

    def is_valid(self):
        pass

    def parseheader(self):
        filebuffer = self._filebuffer
        firstline = filebuffer.readline().decode()

        if not firstline: # for emtpy data files
            return None,-1
        if firstline[0] != '#': # for non-qtlab-like data files
            headerlength = 1
        else: # for qtlab-like data files featuring all kinds of information in python comment lines
            filebuffer.seek(0)
            for i, linebuffer in enumerate(filebuffer):
                line = linebuffer.decode('utf-8')
                if i < 3:
                    continue
                if i > 5:
                    if line[0] != '#': #find the skiprows accounting for the first linebreak in the header
                        headerlength = i
                        break
                if i > 300:
                    break

        filebuffer.seek(0)
        headertext = [next(filebuffer) for x in xrange(headerlength)]
        headertext= b''.join(headertext)
        headertext= headertext.decode('utf-8')
        
        filebuffer.seek(0) #put it back to 0 in case someone else naively reads the filebuffer
        #doregex
        coord_expression = re.compile(r"""                  ^\#\s*Column\s(.*?)\:
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
        coord_expression_short = re.compile(r"""           ^\#\s*Column\s(.*?)\:
                                                            [\r\n]{0,2}
                                                            \#\s*name\:\s(.*?)
                                                            [\r\n]{0,2}
                                                            \#\s*size\:\s(.*?)
                                                            [\r\n]{0,2}
                                                            \#\s*type\:\s(.*?)[\r\n]{0,2}$

                                                            """#annoying \r's...
                                        ,re.VERBOSE |re.MULTILINE)
        val_expression = re.compile(r"""                       ^\#\s*Column\s(.*?)\:
                                                                [\r\n]{0,2}
                                                                \#\s*name\:\s(.*?)
                                                                [\r\n]{0,2}
                                                                \#\s*type\:\s(.*?)[\r\n]{0,2}$
                                                                """
                                            ,re.VERBOSE |re.MULTILINE)
        coord =  coord_expression.findall(headertext)
        val = val_expression.findall(headertext)
        reg_nu = re.compile(r'\{(.*?)\}')
        
        coord = [ zip(('column','end','name','size','start','type'),x) for x in coord]
        coord = [dict(x) for x in coord]
        for i in range(0,len(coord)):
            name_and_unit = reg_nu.findall(coord[i]['name'])
            if not name_and_unit:
                coord[i]['unit'] = ''
                coord[i]['label'] = coord[i]['name']
            else:
                coord[i]['name'] = name_and_unit[0]
                coord[i]['unit'] = name_and_unit[1]
                coord[i]['label'] = name_and_unit[0]

        val = [ zip(('column','name','type'),x) for x in val]
        val = [dict(x) for x in val]
        for i in range(0,len(val)):
            name_and_unit = reg_nu.findall(val[i]['name'])
            if not name_and_unit:
                val[i]['unit'] = ''
                val[i]['label'] = val[i]['name']
            else:
                val[i]['name'] = name_and_unit[0]
                val[i]['unit'] = name_and_unit[1]
                val[i]['label'] = name_and_unit[0]
        header=coord+val

        if not coord: # for data files without the 'start' and 'end' line in the header 
            coord_short = coord_expression_short.findall(headertext)
            coord_short = [ zip(('column','name','size','type'),x) for x in coord_short]
            coord_short = [dict(x) for x in coord_short]
            for i in range(0,len(coord)):
                name_and_unit = reg_nu.findall(coord[i]['name'])
                if not name_and_unit:
                    coord[i]['unit'] = ''
                    coord[i]['label'] = coord[i]['name']
                else:
                    coord[i]['name'] = name_and_unit[0]
                    coord[i]['unit'] = name_and_unit[1]
                    coord[i]['label'] = name_and_unit[0]
                header=coord_short+val
        
        return header,headerlength

def factory_gz_parser(cls):
    # parent class of gz_parser depends on which kind of data file we have
    class gz_parser(cls):
        def __init__(self,filename,filebuffer=None):
            self._file = filename
            
            import gzip
            f = open(self._file,'rb')
            if (f.read(2) == b'\x1f\x8b'):
                f.seek(0)
                gz = super(gz_parser,self).__init__(filename=filename,filebuffer=gzip.GzipFile(fileobj=f))
                return gz
            else:
                #raise Exception('Not a valid gzip file')
                print('Not a valid gzip file')
                gz = super(gz_parser,self).__init__(filename=filename,filebuffer=None)
                return gz
            f.close()
    return gz_parser

#class for supported filetypes, handles which parser class to call
class filetype():
    def __init__(self,filepath=None):
        self._parser = None
        self._filepath = filepath
        self._datparser, self._file_Extension = self.selectparser(filepath)

    def get_parser(self):
        return self._datparser

    @classmethod
    def selectparser(cls,filepath=''):
        cls._SUPPORTED_FILETYPES = ['.dat', '.csv', '.gz']
        cls._SUPPORTED_METATYPES = ['.set', '.txt', '.json', '.csv']
        file_Path, file_Extension = os.path.splitext(filepath)
        # look for supported metadata files
        exts = []
        if file_Extension in cls._SUPPORTED_FILETYPES:
            meta_Extension = []
            for file in os.listdir(os.path.dirname(filepath)):
                if os.path.splitext(file)[1] in cls._SUPPORTED_METATYPES:
                    meta_Extension.append(os.path.splitext(file)[1])
            #if len(meta_Extension) > 1:
            #    print('Too many supported metadata files in measurement folder.')
            #    meta_Extension = []
            if len(meta_Extension) == 0:
                print('No supported metadata extension found. \nOnly ' + ' '.join(cls._SUPPORTED_METATYPES) + ' are supported.')
                meta_Extension = ''
            else:
                meta_Extension = meta_Extension[0]
        else:
            print('Unsupported file extension: ' + file_Extension + '\nOnly ' + ' '.join(cls._SUPPORTED_FILETYPES) + ' are supported.')

        # select correct parser based on metadata extension and data file extension
        if meta_Extension == '.json':
            if file_Extension ==  '.gz':
                file_Path = os.path.splitext(file_Path)[0]
                parser = factory_gz_parser(qcodes_parser)
            elif file_Extension != '.dat':
                print('Wrong file extension for qcodes parser: ' + file_Extension)
                parser = None
            else: 
                parser =  qcodes_parser
        
        elif meta_Extension == '.txt' or meta_Extension == '.csv'  :
            if file_Extension ==  '.gz':
                file_Path = os.path.splitext(file_Path)[0]
                parser = factory_gz_parser(qtm_parser)
            elif file_Extension != '.csv':
                print('Wrong file extension for qtlab parser: ' + file_Extension)
                parser =  None
            else: 
                parser =  qtm_parser
        
        elif meta_Extension == '.set':
            if file_Extension ==  '.gz':
                file_Path = os.path.splitext(file_Path)[0]
                parser = factory_gz_parser(qtlab_parser)
            elif file_Extension != '.dat':
                print('Wrong file extension for qtlab parser: ' + file_Extension)
                parser =  None
            else: 
                parser =  qtlab_parser
        elif file_Extension == '.dat':
            parser = dat_parser
        else:
            print('Error: no parser suitable with combination of data and meta file extensions.')
            parser = None
        return parser, file_Extension

    def getsetfilepath(cls,filepath=''):
        file_Path, file_Extension = os.path.splitext(filepath)
        if file_Extension ==  '.gz':
            file_Path = os.path.splitext(file_Path)[0]
        elif file_Extension != '.dat':
            print('Wrong file extension')
        setfilepath = file_Path + '.set'
        
        if not os.path.exists(setfilepath):
            setfilepath = ''
        
        return setfilepath

    def get_filetype(self):
        return self._file_Extension

        return parser, file_Extension

class Data(pandas.DataFrame):
    
    def __init__(self,*args,**kwargs):
        #args: filepath, sort
        #filepath = kwargs.pop('filepath',None)
        #sort = kwargs.pop('sort',True)

        #dat,header = self.load_file(filepath)
        super(Data,self).__init__(*args,**kwargs)

        self._filepath = None #filepath
        self._header = None #header
        self._unsorted_data = None
        self._sorted_data = None
        self._value_keys = None

    @property
    def _constructor(self):
        return Data

    @classmethod
    def determine_filetype(cls,filepath):
        ftype = filetype(filepath=filepath)
        
        return ftype.get_filetype()

    @classmethod
    def load_header_only(cls,filepath):
        parser = cls.determine_parser(filepath)
        p = parser(filename=filepath,filebuffer=open(filepath,mode='rb'))
        if p._filebuffer is None:
            p = None
            return None
        header,headerlength = p.parseheader()
        df = Data()
        df._header = header
        return df
    
    @classmethod
    def determine_parser(cls,filepath):
        ftype = filetype(filepath=filepath)
        parser = ftype.get_parser()
        
        return parser
    
    @classmethod
    def load_file(cls,filepath):
        parser = cls.determine_parser(filepath)
        p = parser(filename=filepath,filebuffer=open(filepath,mode='rb'))
        
        if p._filebuffer is None:
            p = None
            return None,None
        p.parse()
        return p._data,p._header

    @classmethod
    def from_file(cls, filepath):
        dat,header = cls.load_file(filepath)

        newdataframe = Data(dat)
        newdataframe._header = header

        return newdataframe
    
    @property
    def coordkeys(self):
        coord_keys = [i['name'] for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type']=='coordinate')]
        return coord_keys
    
    @property
    def valuekeys(self):
        value_keys = [i['name'] for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type']=='value')]
        self._value_keys=value_keys
        return value_keys


    @property
    def coordkeys_n(self):
        coord_keys = [i['name'] for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type']=='coordinate')]
        units = [i['unit'] for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type']=='coordinate')]
        labels= [i['label'] for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type']=='coordinate')]
        # for i in range(0,len(self._header)):
        #     if 'label' in self._header[i]:
        #         if self._header[i]['type']=='coordinate':
        #             labels.append(self._header[i]['label'])
        return coord_keys, units, labels
    
    @property
    def valuekeys_n(self):
        value_keys = [i['name'] for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type']=='value')]
        units = [i['unit'] for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type']=='value')]
        labels= [i['label'] for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type']=='value')]
        # for i in range(0,len(self._header)):
        #     if 'label' in self._header[i]:
        #         if self._header[i]['type']=='value':
        #             labels.append(self._header[i]['label'])
        return value_keys, units,labels

    @property
    def sorted_data(self):
        if self._sorted_data is None:
            #sort the data from the last coordinate column backwards
            self._sorted_data = self.sort_values(by=self.coordkeys)
            #self._sorted_data = self._sorted_data.dropna(how='any')
        return self._sorted_data
    
    @property
    def unsorted_data(self):
        if self._unsorted_data is None:
            #sort the data from the last coordinate column backwards
            self._unsorted_data = self#.sort_values(by=self.coordkeys)
            #self._sorted_data = self._sorted_data.dropna(how='any')
        return self._unsorted_data

    @property
    def ndim_sparse(self):
        #returns the amount of columns with more than one unique value in it
        dims = np.array(self.dims)
        nDim = len(dims[dims > 1])
        
        return nDim
    
    @property
    def dims(self):
        #returns an array with the amount of unique values of each coordinate column

        dims = np.array([],dtype='int')
        #first determine the columns belong to the axes (not measure) coordinates
        cols = [i for n,i in enumerate(self._header) if ('column' in self._header[n] and i['type'] == 'coordinate')]

        for i in cols:
            col = getattr(self.sorted_data,i['name'])
            dims = np.hstack( ( dims ,len(col.unique())  ) )

        return dims

    def make_filter_from_uniques_in_columns(self,columns):
    #generator to make a filter which creates measurement 'sets'
        import math
    #arg columns, list of column names which contain the 'uniques' that define a measurement set

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
        xs = self.sorted_data[columns]
        if xs.shape[1] > 1:
            for i in xs.iloc[:,0].unique():
                if math.isnan(i):
                    continue
                for j in self.make_filter_from_uniques_in_columns(columns[1:]):
                    yield (xs.iloc[:,0] == i) & j ## boolean and
        elif xs.shape[1] == 1:
            for i in xs.iloc[:,0].unique():
                if (math.isnan(i)):
                    continue
                yield xs.iloc[:,0] == i
        else:
            #empty list
            yield slice(None) #return a 'semicolon' to select all the values when there's no value to filter on