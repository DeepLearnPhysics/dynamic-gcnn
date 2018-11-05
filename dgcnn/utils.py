from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np

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

def InclusiveGrouping(dist_v,score_v,threshold):
  res = []
  for dist in dist_v:
    np.fill_diagonal(dist,threshold+1)
    cands   = np.where(np.min(dist,axis=0) < threshold)
    grp     = np.zeros(shape=[dist.shape[0]],dtype=np.float32)
    grp[:]  = -1
    latest_grp_id = 0
    # loop over candidates and group coordinates
    friends_v = [np.where(dist[c] < threshold) for c in cands[0]]
    #print('Found',len(cands[0]),'candidates...')
    for i,cand in enumerate(cands[0]):
      if grp[cand]>=0: continue
      # Is any of friend in an existing group? Then use the closest one
      friends = friends_v[i][0]
      grouped_friends = [friend for friend in friends if grp[friend]>=0]
      if len(grouped_friends)>0:
        best_friend = np.argmin(dist[cand][grouped_friends])
        best_friend = grouped_friends[best_friend]
        grp[cand] = grp[best_friend]
        #print('found grouped friends:',grouped_friends)
        #print('setting from best friend',cand,'(dist',dist[cand][best_friend],') grp',grp[cand])
      else:
        grp[friends] = latest_grp_id
        grp[cand] = latest_grp_id
        print('setting',cand,latest_grp_id)
        latest_grp_id +=1
        #print('setting friends',friends,latest_grp_id)

    res.append(grp)
  return res

def ScoreGrouping(dist_v,score_v,threshold):
  res = []
  for idx in range(len(dist_v)):
    dist   = dist_v[idx]
    score  = score_v[idx]
    order  = np.argsort(score * -1.)
    grp    = np.zeros(shape=[dist.shape[0]],dtype=np.float32)
    grp[:] = -1
    new_grp_id = 0
    for i in order:
      pt_idx = order[i]
      if grp[pt_idx] >= 0: continue
      
      if score[pt_idx] > 0.75:
        grp[pt_idx] = new_grp_id
        friends = np.where(dist[pt_idx] < threshold)
        grp[friends] = new_grp_id
        new_grp_id += 1
      else:
        grp_id = -1
        closer_friends = np.argsort(dist[pt_idx])
        for friend in closer_friends:
          if grp[friend] < 0: continue
          grp[pt_idx] = grp[friend]
          break
        if grp[pt_idx]<0:
          grp[pt_idx] = new_grp_id
          friends = np.where(dist[pt_idx] < threshold)
          grp[friends] = new_grp_id
          new_grp_id += 1
    res.append(grp)
  return res

if __name__ == '__main__':
    d=CSVData('aho.csv')
    d.record('acc',1.000001)
    d.record('loss',0.1)
    d.write()
