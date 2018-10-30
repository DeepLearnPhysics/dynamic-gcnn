from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os

class CSVData:

    def __init__(self,fout):
        self._fout = fout
        self._str  = None
        self._dict = {}
        
    def record(self,keys,vals):
        for i,key in enumerate(keys):
            self._dict[key] = vals[i]
        
    def write(self):

        if self._str is None:
            self._fout=open(self._fout,'w')
            self._str=''
            for i,key in enumerate(self._dict.keys()):
                if i:
                    self._fout.write(',')
                    self._str += ','
                self._fout.write(key)
                self._str+='{:f}'
            self._fout.write('\n')
            self._str+='\n'

        self._fout.write(self._str.format(*(self._dict.values())))

    def flush(self):
        if self._fout: self._fout.flush()

    def close(self):
        if self._str is not None:
            self._fout.close()
            
if __name__ == '__main__':
    d=CSVData('aho.csv')
    d.record('acc',1.000001)
    d.record('loss',0.1)
    d.write()
