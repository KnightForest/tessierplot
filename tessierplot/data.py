import pandas
import os
import re
import numpy as np

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
            f = open(self._file)
            self._filebuffer = f
        else:
            f = filebuffer
            self._filebuffer = filebuffer
        self._header,self._headerlength = self.parseheader()

        self._data = pandas.read_csv(f,
                                 sep='\t', \
                                 comment='#',
                                 skiprows=self._headerlength,
                                 names=[i['name'] for i in self._header])
        return super(dat_parser,self).parse()

    def is_valid(self):
        pass
    def parseheader(self):
        filebuffer = self._filebuffer
        for i, line in enumerate(filebuffer):
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
        headertext= ''.join(headertext)
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

        val_expression = re.compile(r"""                       ^\#\s*Column\s(.*?)\:
                                                                [\r\n]{0,2}
                                                                \#\s*name\:\s(.*?)
                                                                [\r\n]{0,2}
                                                                \#\s*type\:\s(.*?)[\r\n]{0,2}$
                                                                """
                                            ,re.VERBOSE |re.MULTILINE)
        coord=  coord_expression.findall(headertext)
        val  = val_expression.findall(headertext)
        coord = [ zip(('column','end','name','size','start','type'),x) for x in coord]
        coord = [dict(x) for x in coord]
        val = [ zip(('column','name','type'),x) for x in val]
        val = [dict(x) for x in val]

        header=coord+val
        return header,headerlength

class gz_parser(dat_parser):
    def __init__(self,filename):
        self._file = filename
        
        import gzip
        f = open(self._file,'rb')
        if (f.read(2) == '\x1f\x8b'):
            f.seek(0)
            gz = super(gz_parser,self).__init__(filename=filename,filebuffer=gzip.GzipFile(fileobj=f))
            return gz
        else:
            raise Exception('Not a valid gzip file')
        
        

class Data(pandas.DataFrame):
    #supported filetypes
    _FILETYPES =   {
        '.dat': dat_parser,
        '.dat.gz': gz_parser
        }

    def __init__(self,*args,**kwargs):
        #args: filepath, sort
        #filepath = kwargs.pop('filepath',None)
        #sort = kwargs.pop('sort',True)

        #dat,header = self.load_file(filepath)
        super(Data,self).__init__(*args,**kwargs)

        self._filepath = None #filepath
        self._header = None #header
        self._sorted_data = None

    @property
    def _constructor(self):
        return Data

    @classmethod
    def determine_filetype(cls,filepath):
        for ext in cls._FILETYPES.keys():
            if filepath.endswith(ext):
                return ext
        return None

    @classmethod
    def load_header_only(cls,filepath):
        parser = cls.determine_parser(filepath)
        header,headerlength = parser(filepath).parseheader()
        df = Data()
        df._header = header
        return df
    
    @classmethod
    def determine_parser(cls,filepath):
        ftype = cls.determine_filetype(filepath)
        if ftype is not None:
            parser = cls._FILETYPES[ftype]
        else:
            raise Exception('Unknown filetype')
        return parser
    
    @classmethod
    def load_file(cls,filepath):
        parser = cls.determine_parser(filepath)
        p = parser(filepath)
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
        coord_keys = [i['name'] for i in self._header if i['type']=='coordinate' ]
        return coord_keys
    @property
    def valuekeys(self):
        value_keys = [i['name'] for i in self._header if i['type']=='value' ]
        return value_keys

    @property
    def sorted_data(self):
        if self._sorted_data is None:
            #sort the data from the last coordinate column backwards
            self._sorted_data = self.sort_values(by=self.coordkeys)
            self._sorted_data = self._sorted_data.dropna(how='any')
        return self._sorted_data

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
        cols = [i for i in self._header if (i['type'] == 'coordinate')]

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
