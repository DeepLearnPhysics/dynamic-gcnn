from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
from sklearn.cluster import DBSCAN


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


def DBSCANGrouping(dist_v, score_v, threshold):
    """
    Using Scikit Learn DBSCAN implementation
    Does not use scores
    """
    labels = []
    # dbscan = DBSCAN(eps=3, min_samples=2, metric='precomputed')
    for dist in dist_v:
        db = DBSCAN(eps=threshold, min_samples=1, metric='precomputed').fit(dist)
        print('Clusters: ', np.unique(db.labels_))
        labels.append(db.labels_.astype(np.float32))
    return labels


def SGPNGrouping(dist_v, scores_v, threshold, debug=False):
    """
    Grouping method from the paper SGPN.
    Discards groups with a low cardinal or low score, then NMS-style pruning.
    /!\ Other thresholds hardcoded here!
    """
    assert len(dist_v) == len(scores_v)
    threshold_dist = threshold
    threshold_score = 0.2
    threshold_cardinal = 20
    threshold_iou = 0.05
    results = []
    for dist, scores in zip(dist_v, scores_v):
        if debug: print(scores.mean(), scores.std())
        # First create proposals based on similitude distance
        proposals = dist < threshold_dist

        # Discard proposals with small confidence score
        keep_index = scores > threshold_score
        proposals = proposals[keep_index, :]
        scores = scores[keep_index]
        if debug: print('score', proposals.shape, scores.shape)

        # Discard proposals with small cardinal
        keep_index2 = np.sum(proposals, axis=1) > threshold_cardinal
        proposals = proposals[keep_index2, :]
        scores = scores[keep_index2]
        if debug: print('cardinal', proposals.shape, scores.shape)
        # IoU NMS
        # Order by score
        index = np.argsort(scores)
        scores = scores[index]
        proposals = proposals[index, :]

        final_groups = []
        while proposals.shape[0] > 1:
            intersection = np.sum(np.logical_and(proposals[0], proposals), axis=-1)
            union = np.sum(np.logical_or(proposals[0], proposals), axis=-1) + 1e-6
            iou = intersection / union
            # print('iou', np.mean(iou))
            groups_to_merge = iou > threshold_iou
            new_group = np.logical_or.reduce(proposals[groups_to_merge])
            final_groups.append(new_group)
            proposals = proposals[iou <= threshold_iou]

        final_groups = np.array(final_groups)
        if final_groups.shape[0]:
            final_groups = np.argmax(final_groups, axis=0)
        else:  # No group left
            final_groups = np.zeros((dist.shape[1],))
        results.append(final_groups.astype(np.float32))
    return results


if __name__ == '__main__':
    d=CSVData('aho.csv')
    d.record('acc',1.000001)
    d.record('loss',0.1)
    d.write()
