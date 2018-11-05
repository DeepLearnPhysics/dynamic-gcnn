from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import dgcnn
import tensorflow as tf
  
class trainval(object):

  def __init__(self,flags):
    self._flags = flags

  def initialize(self):

    self._ops = {}

    with tf.device('/cpu:0'):
      self._optimizer = tf.train.AdamOptimizer(self._flags.LEARNING_RATE)

      self._feed_blob = {}
      self._points_v  = []
      self._grp_v  = []
      self._pdg_v  = []
      self._alpha  = []
      self._group_pred_v = []
      self._group_conf_v = []
      loss_same_group_v = []
      loss_same_pdg_v   = []
      loss_diff_group_v = []
      loss_cluster_v    = []
      loss_conf_v  = []
      loss_total_v = []
      gradient_v   = []
      for i, gpu_id in enumerate(self._flags.GPUS):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('GPU%d' % gpu_id) as scope:
            with tf.variable_scope("dgcnn",reuse=tf.AUTO_REUSE):
              points = tf.placeholder(tf.float32, 
                                      shape=(self._flags.MINIBATCH_SIZE,None,self._flags.NUM_CHANNEL))
              grp_id = tf.placeholder(tf.float32,shape=(self._flags.MINIBATCH_SIZE,None,1))
              pdg_id = tf.placeholder(tf.float32,shape=(self._flags.MINIBATCH_SIZE,None,1))
              alpha  = tf.placeholder(tf.float32,shape=())
              self._points_v.append(points)
              self._grp_v.append(grp_id)
              self._pdg_v.append(pdg_id)
              self._alpha.append(alpha)
              
              num_point = tf.shape(self._points_v[-1])[1]
              grp_dist = dgcnn.ops.dist_nn(grp_id)
              pdg_dist = dgcnn.ops.dist_nn(pdg_id)
              # sim_grp0 is (B,N,N), 1.0 for point combinations belonging to the same group (diagonal is 1.0)
              label_same_group = tf.cast((grp_dist < 0.5),tf.float32)
              # sim_grp1 is (B,N,N), 1.0 for point combinations belonging to the same PDG but different group (but diagonal is 0.0)
              label_same_pdg   = tf.cast(tf.logical_and((pdg_dist < 0.5), (grp_dist > 0.5)),tf.float32)
              # sim_grp2 is (B,N,N), 1.0 for point combinations belonging to the different PDG and different group (but diagonal is 0.0)
              label_diff_group = tf.cast((pdg_dist > 0.5),tf.float32)
              # construct a model and get the output: similarity matrix (B,N,N) and confidence array (B,N)
              pred, conf = dgcnn.model.build(points, self._flags)
              self._group_pred_v.append(pred)
              #
              # Compute loss for similarity matrix
              #
              num_same_group = (tf.reduce_sum(label_same_group) - tf.cast(num_point,tf.float32))#/2.
              num_same_pdg   = tf.reduce_sum(label_same_pdg)# / 2.
              num_diff_group = tf.reduce_sum(label_diff_group)# / 2.
              loss_same_group = tf.cond(num_same_group > 0,
                                        lambda: tf.reduce_sum(pred * label_same_group) / num_same_group,
                                        lambda: tf.constant(0.0))
              loss_same_pdg   = tf.cond(num_same_pdg > 0,
                                        lambda: tf.reduce_sum(tf.maximum(self._flags.K1 - pred,0.0) * label_same_pdg) / num_same_pdg * alpha,
                                        lambda: tf.constant(0.0))
              loss_diff_group = tf.cond(num_diff_group > 0,
                                        lambda: tf.reduce_sum(tf.maximum(self._flags.K2 - pred,0.0) * label_diff_group) / num_diff_group,
                                        lambda: tf.constant(0.0))
              loss_cluster = loss_same_group + loss_same_pdg + loss_diff_group
              loss_same_group_v.append(loss_same_group)
              loss_same_pdg_v.append(loss_same_pdg)
              loss_diff_group_v.append(loss_diff_group)
              loss_cluster_v.append(loss_cluster)
              #
              # Compute loss for confidence score
              #
              conf = tf.nn.sigmoid(conf)
              label_same_group = (grp_dist < 0.5)
              pred_same_group  = (pred < self._flags.K1)
              conf_label_numerator   = tf.reduce_sum(tf.cast(tf.logical_and (pred_same_group,label_same_group),tf.float32),axis=2)
              conf_label_denominator = tf.reduce_sum(tf.cast(tf.logical_or  (pred_same_group,label_same_group),tf.float32),axis=2) + 1.e-6
              conf_label = conf_label_numerator / conf_label_denominator
              loss_conf = tf.reduce_mean(tf.squared_difference(conf, conf_label))
              self._group_conf_v.append(conf)
              loss_conf_v.append(loss_conf)

              # total loss
              loss_total = loss_conf + loss_cluster
              loss_total_v.append(loss_total)
              if self._flags.TRAIN:
                grad = self._optimizer.compute_gradients(loss_total)
                gradient_v.append(grad)

      # Average loss across GPUs
      self._loss_same_group = tf.add_n(loss_same_group_v) / float(len(self._flags.GPUS))
      self._loss_same_pdg   = tf.add_n(loss_same_pdg_v  ) / float(len(self._flags.GPUS))
      self._loss_diff_group = tf.add_n(loss_diff_group_v) / float(len(self._flags.GPUS))
      self._loss_cluster    = tf.add_n(loss_cluster_v   ) / float(len(self._flags.GPUS))
      self._loss_conf       = tf.add_n(loss_conf_v      ) / float(len(self._flags.GPUS))
      self._loss_total      = tf.add_n(loss_total_v     ) / float(len(self._flags.GPUS))
      # If training, average gradients across GPUs
      if self._flags.TRAIN:
        average_gradient_v = []
        for grad_and_var_v in zip(*gradient_v):
          v = []
          for g, _ in grad_and_var_v:
            v.append(tf.expand_dims(g,0))
          
          grad = tf.reduce_mean(tf.concat(v,0), 0)
          
          if self._flags.DEBUG:
            print('Computing gradients for %s from %d GPUs' % (grad_and_var_v[0][1].name,len(grad_and_var_v)))
          average_gradient_v.append((grad, grad_and_var_v[0][1]))
      
        accum_vars   = [tf.Variable(v.initialized_value(),trainable=False) for v in tf.trainable_variables()]
        self._zero_grad    = [v.assign(tf.zeros_like(v)) for v in accum_vars]
        self._accum_grad_v = []

        self._accum_grad_v += [accum_vars[j].assign_add(g[0]) for j,g in enumerate(average_gradient_v)]
        self._apply_grad = self._optimizer.apply_gradients(zip(accum_vars, tf.trainable_variables()))

        # Merge summary
        tf.summary.scalar('loss_same_group', self._loss_same_group)
        tf.summary.scalar('loss_same_pdg',   self._loss_same_pdg  )
        tf.summary.scalar('loss_diff_group', self._loss_diff_group)
        tf.summary.scalar('loss_cluster',    self._loss_cluster   )
        tf.summary.scalar('loss_conf',       self._loss_conf      )
        tf.summary.scalar('loss_total',      self._loss_total     )
        self._merged_summary=tf.summary.merge_all()
      
  def feed_dict(self,data,grp=None,pdg=None,alpha=None):
    res = {}
    for i,gpu_id in enumerate(self._flags.GPUS):
      res[self._points_v [i]] = data [i]
      if grp:   res[self._grp_v[i]] = grp  [i]
      if pdg:   res[self._pdg_v[i]] = pdg  [i]
      if alpha: res[self._alpha[i]] = alpha
    return res

  def make_summary(self, sess, data, label, weight, alpha=1.0):
    if not self._flags.TRAIN:
      raise NotImplementedError    
    feed_dict = self.feed_dict(data,label,weight,alpha)
    return sess.run(self._merged_summary,feed_dict=feed_dict)
  
  def inference(self,sess,data,grp=None,pdg=None,alpha=None):
    feed_dict = self.feed_dict(data,grp,pdg,alpha)
    ops  = list(self._group_pred_v)
    ops += list(self._group_conf_v)
    if grp is not None and pdg is not None:
      ops += [self._loss_same_group,self._loss_same_pdg,self._loss_diff_group,self._loss_cluster]
      ops += [self._loss_conf,self._loss_total]
    return sess.run(ops, feed_dict = feed_dict)
    
  def accum_gradient(self, sess, data, grp, pdg, alpha=1.0, summary=False):
    if not self._flags.TRAIN:
      raise NotImplementedError
    feed_dict = self.feed_dict(data,grp,pdg,alpha)
    ops  = [self._accum_grad_v, self._loss_same_group, self._loss_same_pdg, self._loss_diff_group, self._loss_cluster]
    ops += [self._loss_conf, self._loss_total]
    if summary:
      ops += [self._merged_summary]
    return sess.run(ops, feed_dict = feed_dict)

  def zero_gradients(self, sess):
    if not self._flags.TRAIN:
      raise NotImplementedError
    return sess.run([self._zero_grad])
    
  def apply_gradient(self,sess):
    if not self._flags.TRAIN:
      raise NotImplementedError
    return sess.run(self._apply_grad)
