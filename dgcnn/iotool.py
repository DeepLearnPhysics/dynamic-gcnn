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
        self._data   = None
        self._label  = None
        self._weight = None
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
        ch_data   = TChain('sparse3d_%s_tree' % self._flags.DATA_KEY)
        ch_label  = None
        ch_weight = None
        if self._flags.LABEL_KEY:
            ch_label  = TChain('sparse3d_%s_tree' % self._flags.LABEL_KEY)
        if self._flags.WEIGHT_KEY:
            ch_weight = TChain('sparse3d_%s_tree' % self._flags.WEIGHT_KEY)
        for f in self._flags.INPUT_FILE:
            ch_data.AddFile(f)
            if ch_label:  ch_label.AddFile(f)
            if ch_weight: ch_weight.AddFile(f)
        self._data   = []
        self._label  = []
        self._weight = []
        br_data,br_label,br_weight=(None,None,None)
        event_fraction = 1./ch_data.GetEntries() * 100.
        total_point = 0.
        for i in range(ch_data.GetEntries()):
            ch_data.GetEntry(i)
            if ch_label:  ch_label.GetEntry(i)
            if ch_weight: ch_weight.GetEntry(i)
            if br_data is None:
                br_data  = getattr(ch_data, 'sparse3d_%s_branch' % self._flags.DATA_KEY)
                if ch_label:  br_label  = getattr(ch_label, 'sparse3d_%s_branch' % self._flags.LABEL_KEY)
                if ch_weight: br_weight = getattr(ch_weight,'sparse3d_%s_branch' % self._flags.WEIGHT_KEY)
            num_point = br_data.as_vector().size()
            if num_point < 256: continue
            print(br_data)
            np_data  = np.zeros(shape=(num_point,4),dtype=np.float32)
            print(np_data.shape)
            larcv.fill_3d_pcloud(br_data,  np_data)
            print(np_data.shape)
            self._data.append(np_data)
            self._event_keys.append((br_data.run(),br_data.subrun(),br_data.event()))
            self._metas.append(larcv.Voxel3DMeta(br_data.meta()))
            if ch_label:
                np_label = np.zeros(shape=(num_point,1),dtype=np.float32)
                print(np_label.shape)
                larcv.fill_3d_pcloud(br_label, np_label)
                np_label = np_label.reshape([num_point]) - 1.
                self._label.append(np_label)
            if ch_weight:
                np_weight = np.zeros(shape=(num_point,1),dtype=np.float32)
                print(np_weight.shape)
                larcv.fill_3d_pcloud(br_weight, np_weight)
                np_weight = np_weight.reshape([num_point])
                np_weight = np_weight / np_weight.sum() * len(np_weight)
                self._weight.append(np_weight)
            total_point += np_data.size
            sys.stdout.write('Processed %d%% ... %d MB\r' % (int(event_fraction*i),int(total_point*4*2/1.e6)))
            sys.stdout.flush()

        sys.stdout.write('\n')
        sys.stdout.flush()
        self._num_channels = self._data[-1].shape[-1]
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
        data,label,weight=(None,None,None)
        start,end=(-1,-1)
        if self._flags.SHUFFLE:
            start = int(np.random.random() * (self.num_entries() - self.batch_size()))
            end   = start + self.batch_size()
            idx   = np.arange(start,end,1)
            data = self._data[start:end]
            if len(self._label)  > 0: label  = self._label[start:end]
            if len(self._weight) > 0: weight = self._weight[start:end]
        else:
            start = self._last_entry+1
            end   = start + self.batch_size()
            if end < self.num_entries():
                idx = np.arange(start,end,1)
                data = self._data[start:end]
                if len(self._label)  > 0: label  = self._label[start:end]
                if len(self._weight) > 0: weight = self._weight[start:end]
            else:
                idx = np.concatenate([np.arange(start,self.num_entries(),1),np.arange(0,end-self.num_entries(),1)])
                data = self._data[start:] + self._data[0:end-self.num_entries()]
                if len(self._label)  > 0: label  = self._label[start:]  + self._label[0:end-self.num_entries()]
                if len(self._weight) > 0: weight = self._weight[start:] + self._weight[0:end-self._num_entries()]
        self._last_entry = idx[-1]

        return idx,data,label,weight

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
        data = self._data[idx]
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
        
        if len(self._label) > 0:
            label = self._label[idx]
            label = label.astype(np.float32).reshape([len(label),1])
            label = np.concatenate([pos,label],axis=1)
            larcv_label = self._fout.get_data('sparse3d','label')
            vs = larcv.as_tensor3d(label,meta,-1.)
            larcv_label.set(vs,meta)

        self._fout.set_id(keys[0],keys[1],keys[2])
        self._fout.save_entry()
        
    def finalize(self):
        if self._fout:
            self._fout.finalize()
            
class io_h5(io_base):

    def __init__(self,flags):
        super(io_h5,self).__init__(flags=flags)
        self._flags  = flags
        self._data   = None
        self._label  = None
        self._weight = None
        self._fout   = None
        self._ohandler_data = None
        self._ohandler_label = None
        self._ohandler_softmax = None
        self._has_label = False

    def initialize(self):
        self._last_entry = -1
        # Prepare input
        import h5py as h5
        self._data   = None
        self._label  = None
        self._weight = None
        for f in self._flags.INPUT_FILE:
            f = h5.File(f,'r')
            if self._data is None:
                self._data  = np.array(f[self._flags.DATA_KEY ])
                if self._flags.LABEL_KEY : self._label  = np.array(f[self._flags.LABEL_KEY])
                if self._flags.WEIGHT_KEY: self._weight = np.array(f[self._flags.WEIGHT_KEY])
            else:
                self._data  = np.concatenate(self._data, np.array(f[self._flags.DATA_KEY ]))
                if self._label  : self._label  = np.concatenate(self._label, np.array(f[self._flags.LABEL_KEY ]))
                if self._weight : self._weight = np.concatenate(self._weight,np.array(f[self._flags.WEIGHT_KEY]))
        self._num_channels = self._data[-1].shape[-1]
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
    def store(self,idx,softmax):
        if self._fout is None:
            raise NotImplementedError
        idx=int(idx)
        if idx >= self.num_entries():
            raise ValueError
        data = self._data[idx]
        self._ohandler_data.append(data[None])
        self._ohandler_softmax.append(softmax[None])
        if self._label is not None:
            label = self._label[idx]
            self._ohandler_label.append(label[None])

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
        data = self._data[idx, ...]
        label,weight=(None,None)
        if self._label  : label  = self._label[idx, ...]
        if self._weight : weight = self._weight[idx, ...]
        return idx, data, label, weight

    def finalize(self):
        if self._fout:
            self._fout.close()

def io_factory(flags):
    if flags.IO_TYPE == 'h5':
        return io_h5(flags)
    if flags.IO_TYPE == 'larcv':
        return io_larcv(flags)
    raise NotImplementedError
