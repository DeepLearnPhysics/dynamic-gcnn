from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
class io_base(object):

    def __init__(self,flags):
        self._batch_size = flags.BATCH_SIZE
   
    def batch_size(self,size=None):
        if size is None: return self._batch_size
        self._batch_size = int(size)

    def num_entries(self):
        raise NotImplementedError

    def initialize(self):
        raise NotImplementedError

    def store(self):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

class io_larcv(io_base):

    def __init__(self,flags):
        super(io_larcv,self).__init__(flags=flags)
        self._flags = flags
        self._data  = None
        self._label = None
        self._fout  = None
        self._num_entries = None

    def num_entries(self):
        return self._num_entries

    def initialize(self):
        # configure the input
        from larcv import larcv
        from ROOT import TChain
        ch_data  = TChain('sparse3d_%s_tree' % self._flags.DATA_KEY)
        ch_label = None
        if self._flags.LABEL_KEY:
            ch_label = TChain('sparse3d_%s_tree' % self._flags.LABEL_KEY)
        for f in self._flags.INPUT_FILE:
            ch_data.AddFile(f)
            if ch_label: ch_label.AddFile(f)
        self._data  = []
        self._label = []
        br_data,br_label=(None,None)
        event_fraction = 1./ch_data.GetEntries() * 100.
        total_point = 0.
        for i in range(ch_data.GetEntries()):
            ch_data.GetEntry(i)
            if ch_label: ch_label.GetEntry(i)
            if br_data is None:
                br_data  = getattr(ch_data, 'sparse3d_%s_branch' % self._flags.DATA_KEY)
                if ch_label: br_label = getattr(ch_label,'sparse3d_%s_branch' % self._flags.LABEL_KEY)
            num_point = br_data.as_vector().size()
            if num_point < 256: continue
            np_data  = np.zeros(shape=(num_point,4),dtype=np.float32)
            larcv.fill_3d_pcloud(br_data,  np_data)
            self._data.append(np_data)
            if ch_label:
                np_label = np.zeros(shape=(num_point,1),dtype=np.float32)
                larcv.fill_3d_pcloud(br_label, np_label)
                np_label = np_label.reshape([num_point]) - 1.
                self._label.append(np_label)
            total_point += np_data.size
            sys.stdout.write('Processed %d%% ... %d MB\r' % (int(event_fraction*i),int(total_point*4*2/1.e6)))
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()
        self._num_entries = len(self._data)
        # Output
        if self._flags.OUTPUT_FILE:
            import tempfile
            cfg = '''
IOManager: {
      Verbosity:   2
      Name:        "IOManager"
      IOMode:      1
      OutFileName: "%s"
      InputFiles:  []
      InputDirs:   []
      StoreOnlyType: []
      StoreOnlyName: []
    }
                  '''
            cfg = cfg % self._flags.OUTPUT_FILE
            cfg_file = tempfile.NamedTemporaryFile('w')
            cfg_file.write(cfg)
            cfg_file.flush()
            self._fout = larcv.IOManager(cfg_file.name)
            self._fout.initialize()
            
    def next(self):
        start = int(np.random.random() * (self.num_entries() - self.batch_size()))
        end   = start + self.batch_size()
        idx   = np.arange(start,end,1)
        if len(self._label) < 1:
            return self._data[start:end], None, idx
        return self._data[start:end], self._label[start:end], idx

    def store(self,data,softmax,label=None):
        raise NotImplementedError

    def finalize(self):
        if self._fout:
            self._fout.finalize()
            
class io_h5(io_base):

    def __init__(self,flags):
        super(io_h5,self).__init__(flags=flags)
        self._flags = flags
        self._data  = None
        self._label = None
        self._fout  = None
        self._ohandler_data = None
        self._ohandler_label = None
        self._ohandler_softmax = None
        self._num_entries = None

    def num_entries(self):
        return self._num_entries

    def initialize(self):
        # Prepare input
        import h5py as h5
        self._data  = None
        self._label = None
        for f in self._flags.INPUT_FILE:
            f = h5.File(f,'r')
            if self._data is None:
                self._data  = np.array(f[self._flags.DATA_KEY ])
                if self._flags.LABEL_KEY: self._label = np.array(f[self._flags.LABEL_KEY])
            else:
                self._data  = np.concatenate(self._data, np.array(f[self._flags.DATA_KEY ]))
                if self._label: self._label = np.concatenate(self._label,np.array(f[self._flags.LABEL_KEY]))
        self._num_entries = len(self._data)
        # Prepare output
        if self._flags.OUTPUT_FILE:
            import tables
            FILTERS = tables.Filters(complib='zlib', complevel=5)
            self._fout = tables.open_file(self._flags.OUTPUT_FILE,mode='w', filters=FILTERS)
            data_shape = list(self._data[0].shape)
            data_shape.insert(0,0)
            self._ohandler_data = self._fout.create_earray(self._fout.root,self._flags.DATA_KEY,tables.Float32Atom(),shape=data_shape)
            self._ohandler_softmax = self._fout.create_earray(self._fout.root,'softmax',tables.Float32Atom(),shape=data_shape)
            if self._label:
                data_shape = list(self._label[0].shape)
                data_shape.insert(0,0)
                self._ohandler_label = self._fout.create_earray(self._fout.root,self._flags.LABEL_KEY,tables.Float32Atom(),shape=data_shape)
            
    def store(self,data,softmax,label=None):
        if self._fout is None:
            raise NotImplementedError
        self._ohandler_data.append(data[None])
        self._ohandler_softmax.append(softmax[None])
        if label:
            self._ohandler_label.append(label[None])

    def next(self):
        idx = np.arange(self.num_entries())
        np.random.shuffle(idx)
        idx = idx[0:self.batch_size()]
        if self._label is None:
            return self._data[idx, ...], None, idx
        return self._data[idx, ...], self._label[idx, ...], idx

    def finalize(self):
        if self._fout:
            self._fout.close()

def io_factory(flags):
    if flags.IO_TYPE == 'h5':
        return io_h5(flags)
    if flags.IO_TYPE == 'larcv':
        return io_larcv(flags)
    raise NotImplementedError
