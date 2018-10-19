from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import sys
class io_base(object):

    def __init__(self,flags):
        self._batch_size   = flags.BATCH_SIZE
        self._num_entries  = -1
        self._num_channels = -1
   
    def batch_size(self,size=None):
        if size is None: return self._batch_size
        self._batch_size = int(size)

    def num_entries(self):
        return self._num_entries

    def num_channels(self):
        return self._num_channels

    def initialize(self):
        raise NotImplementedError

    def store(self,idx,softmax):
        raise NotImplementedError

    def next(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError

class io_larcv(io_base):

    def __init__(self,flags):
        super(io_larcv,self).__init__(flags=flags)
        self._flags  = flags
        self._blob   = {}
        self._fout   = None
        self._last_entry = -1
        self._event_keys = []
        self._metas      = []

    def initialize(self):
        self._last_entry = -1
        self._event_keys = []
        self._metas = []
        # configure the input
        from larcv import larcv
        from ROOT import TChain
        ch_blob = {}
        br_blob = {}
        for key in self._flags.DATA_KEYS:
            ch_blob[key]=TChain('sparse3d_%s_tree' % key)
            self._blob[key]=[]
        for f in self._flags.INPUT_FILE:
            for ch in ch_blob.values():
                ch.AddFile(f)
        ach = ch_blob.values()[0]
        event_fraction = 1./ach.GetEntries() * 100.
        total_data = 0.
        for i in range(ach.GetEntries()):
            for key,ch in ch_blob.iteritems():
                ch.GetEntry(i)
                if i == 0:
                    br_blob[key] = getattr(ch, 'sparse3d_%s_branch' % key)
            num_point = br_blob.values()[0].as_vector().size()
            if num_point < self._flags.KVALUE: continue

            # special treatment for the 1st data
            br_data = br_blob[self._flags.DATA_KEYS[0]]
            np_data  = np.zeros(shape=(num_point,4),dtype=np.float32)
            larcv.fill_3d_pcloud(br_data, np_data)
            self._blob[self._flags.DATA_KEYS[0]].append(np_data)
            self._event_keys.append((br_data.run(),br_data.subrun(),br_data.event()))
            self._metas.append(larcv.Voxel3DMeta(br_data.meta()))
            # for the rest, different treatment
            for key in self._flags.DATA_KEYS[1:]:
                br = br_blob[key]
                np_data = np.zeros(shape=(num_point,1),dtype=np.float32)
                larcv.fill_3d_pcloud(br,np_data)
                self._blob[key].append(np_data)
            total_data += num_point * 4 * (4 + len(self._flags.DATA_KEYS)-1)
            sys.stdout.write('Processed %d%% ... %d MB\r' % (int(event_fraction*i),int(total_data/1.e6)))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()
        data = self._blob[self._flags.DATA_KEYS[0]]
        self._num_channels = data[0].shape[-1]
        self._num_entries = len(data)
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
        blob = {}
        start,end=(-1,-1)
        if self._flags.SHUFFLE:
            start = int(np.random.random() * (self.num_entries() - self.batch_size()))
            end   = start + self.batch_size()
            idx   = np.arange(start,end,1)
            for key,val in self._blob.iteritems():
                blob[key] = val[start:end]
        else:
            start = self._last_entry+1
            end   = start + self.batch_size()
            if end < self.num_entries():
                idx = np.arange(start,end,1)
                for key,val in self._blob.iteritems():
                    blob[key] = val[start:end]
            else:
                idx = np.concatenate([np.arange(start,self.num_entries(),1),np.arange(0,end-self.num_entries(),1)])
                for key,val in self._blob.iteritems():
                    blob[key] = val[start:] + val[0:end-self.num_entries()]
        self._last_entry = idx[-1]

        return idx,blob

    def store(self,idx,softmax):
        from larcv import larcv
        if self._fout is None:
            raise NotImplementedError
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        keys = self._event_keys[idx]
        meta = self._metas[idx]
        
        larcv_data = self._fout.get_data('sparse3d',self._flags.DATA_KEY)
        data = self._blob[self._flags.DATA_KEYS[0]][idx]
        vs = larcv.as_tensor3d(data,meta,0.)
        larcv_data.set(vs,meta)

        pos = data[:,0:3]
        score = np.max(softmax,axis=1).reshape([len(softmax),1])
        score = np.concatenate([pos,score],axis=1)
        prediction = np.argmax(softmax,axis=1).astype(np.float32).reshape([len(softmax),1])
        prediction = np.concatenate([pos,prediction],axis=1)
        
        larcv_softmax = self._fout.get_data('sparse3d','softmax')
        vs = larcv.as_tensor3d(score,meta,-1.)
        larcv_softmax.set(vs,meta)

        larcv_prediction = self._fout.get_data('sparse3d','prediction')
        vs = larcv.as_tensor3d(prediction,meta,-1.)
        larcv_prediction.set(vs,meta)

        for key,val in self._flags.DATA_KEYS:
            if key == self._flags.DATA_KEYS[0]: continue
            data = val[idx]
            data = data.astype(np.float32).reshape([len(data),1])
            data = np.concatenate([pos,data],axis=1)
            larcv_data = self._fout.get_data('sparse3d',key)
            vs = larcv.as_tensor3d(data,meta,-1.)
            larcv_data.set(vs,meta)

        self._fout.set_id(keys[0],keys[1],keys[2])
        self._fout.save_entry()
        
    def finalize(self):
        if self._fout:
            self._fout.finalize()
            
class io_h5(io_base):

    def __init__(self,flags):
        super(io_h5,self).__init__(flags=flags)
        self._flags  = flags
        self._blob   = {}
        self._fout   = None
        self._ohandler_data = None
        self._ohandler_label = None
        self._ohandler_softmax = None
        self._has_label = False

    def initialize(self):
        self._last_entry = -1
        # Prepare input
        import h5py as h5
        self._blob   = {}
        for f in self._flags.INPUT_FILE:
            f = h5.File(f,'r')
            if len(self._blob) == 0:
                for key in self._flags.DATA_KEYS:
                    self._blob[key] = np.array(f[key])
            else:
                for key in self._flags.DATA_KEYS:
                    self._blob[key] = np.concatenate(self._blob[key],np.array(f[key]))
        data = self._blob[self.DATA_KEYS[0]]
        self._num_channels = data[-1].shape[-1]
        self._num_entries = len(data)
        # Prepare output
        if self._flags.OUTPUT_FILE:
            import tables
            FILTERS = tables.Filters(complib='zlib', complevel=5)
            self._fout = tables.open_file(self._flags.OUTPUT_FILE,mode='w', filters=FILTERS)
            data_shape = list(data[0].shape)
            data_shape.insert(0,0)
            self._ohandler_data = self._fout.create_earray(self._fout.root,self._flags.DATA_KEY,tables.Float32Atom(),shape=data_shape)
            self._ohandler_softmax = self._fout.create_earray(self._fout.root,'softmax',tables.Float32Atom(),shape=data_shape)
    def store(self,idx,softmax):
        if self._fout is None:
            raise NotImplementedError
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        data = self._blob[self._flags.DATA_KEYS[0]][idx]
        self._ohandler_data.append(data[None])
        self._ohandler_softmax.append(softmax[None])

    def next(self):
        idx = None
        if self._flags.SHUFFLE:
            idx = np.arange(self.num_entries())
            np.random.shuffle(idx)
            idx = idx[0:self.batch_size()]
        else:
            start = self._last_entry+1
            end   = start + self.batch_size()
            if end < self.num_entries():
                idx = np.arange(start,end)
            else:
                idx = np.concatenate([np.arange(start,self.num_entries()),np.arange(0,end-self.num_entries())])
        self._last_entry = idx[-1]
        blob = {}
        for key,val in self._blob.iteritems():
            blob[key] = val[idx, ...]
        return idx, blob

    def finalize(self):
        if self._fout:
            self._fout.close()

def io_factory(flags):
    if flags.IO_TYPE == 'h5':
        return io_h5(flags)
    if flags.IO_TYPE == 'larcv':
        return io_larcv(flags)
    raise NotImplementedError
