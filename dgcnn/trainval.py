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
      self._points_v   = []
      self._grp_v = []
      self._pdg_v = []
      self._alpha = []
      self._pred_v= []
      loss0_v     = []
      loss1_v     = []
      loss2_v     = []
      losstotal_v = []
      grad_v      = []
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
              sim_grp0 = tf.cast((grp_dist < 0.5),tf.float32)
              sim_grp1 = tf.cast(tf.logical_and((pdg_dist < 0.5), (grp_dist > 0.5)),tf.float32)
              sim_grp2 = tf.cast((pdg_dist > 0.5),tf.float32)
              pred = dgcnn.model.build(points, self._flags)
              self._pred_v.append(pred)
              # If training, compute gradients
              num_sim_grp0 = (tf.reduce_sum(sim_grp0) - tf.cast(num_point,tf.float32))#/2.
              num_sim_grp1 = tf.reduce_sum(sim_grp1)# / 2.
              num_sim_grp2 = tf.reduce_sum(sim_grp2)# / 2.
              self._num_point = num_point
              self._num_sim_grp0 = num_sim_grp0
              self._num_sim_grp1 = num_sim_grp1
              self._num_sim_grp2 = num_sim_grp2
              self._elements = tf.maximum((1.0 - (sim_grp1 * pred)),0.0)
              loss0 = tf.cond(num_sim_grp0 > 0,
                              lambda: tf.reduce_sum(pred * sim_grp0) / num_sim_grp0,
                              lambda: tf.constant(0.0))
              loss1 = tf.cond(num_sim_grp1 > 0,
                              lambda: tf.reduce_sum(tf.maximum(self._flags.K1 - pred,0.0) * sim_grp1) / num_sim_grp1 * alpha,
                              lambda: tf.constant(0.0))
              loss2 = tf.cond(num_sim_grp2 > 0,
                              lambda: tf.reduce_sum(tf.maximum(self._flags.K2 - pred,0.0) * sim_grp2) / num_sim_grp2,
                              lambda: tf.constant(0.0))
              #loss0 = tf.reduce_sum(pred * sim_grp0) / tf.cast(num_point,tf.float32)
              #loss1 = tf.reduce_sum(1. / ((sim_grp1 * pred) + 100000.)) / tf.cast(num_point,tf.float32)
              #loss2 = tf.reduce_sum(1. / ((sim_grp2 * pred) + 100000.)) / tf.cast(num_point,tf.float32)
              loss_total = loss0 + loss1 + loss2
              loss0_v.append(loss0)
              loss1_v.append(loss1)
              loss2_v.append(loss2)
              losstotal_v.append(loss_total)
              if self._flags.TRAIN:
                grad = self._optimizer.compute_gradients(loss_total)
                grad_v.append(grad)

      # Average loss across GPUs
      self._loss0 = tf.add_n(loss0_v) / float(len(self._flags.GPUS))
      self._loss1 = tf.add_n(loss1_v) / float(len(self._flags.GPUS))
      self._loss2 = tf.add_n(loss2_v) / float(len(self._flags.GPUS))
      self._losstotal = tf.add_n(losstotal_v) / float(len(self._flags.GPUS))
                
      # If training, average gradients across GPUs
      if self._flags.TRAIN:
        average_grad_v = []
        for grad_and_var_v in zip(*grad_v):
          v = []
          for g, _ in grad_and_var_v:
            v.append(tf.expand_dims(g,0))
          
          grad = tf.reduce_mean(tf.concat(v,0), 0)
          
          if self._flags.DEBUG:
            print('Computing gradients for %s from %d GPUs' % (grad_and_var_v[0][1].name,len(grad_and_var_v)))
          average_grad_v.append((grad, grad_and_var_v[0][1]))
      
        accum_vars   = [tf.Variable(v.initialized_value(),trainable=False) for v in tf.trainable_variables()]
        self._zero_grad    = [v.assign(tf.zeros_like(v)) for v in accum_vars]
        self._accum_grad_v = []

        self._accum_grad_v += [accum_vars[j].assign_add(g[0]) for j,g in enumerate(average_grad_v)]
        self._apply_grad = self._optimizer.apply_gradients(zip(accum_vars, tf.trainable_variables()))

        # Merge summary
        tf.summary.scalar('loss', self._losstotal)
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
    ops = list(self._pred_v)
    if grp is not None and pdg is not None:
      ops += [self._loss0,self._loss1,self._loss2,self._losstotal]
    return sess.run(ops, feed_dict = feed_dict)
    
  def accum_gradient(self, sess, data, grp, pdg, alpha=1.0, summary=False):
    if not self._flags.TRAIN:
      raise NotImplementedError
    feed_dict = self.feed_dict(data,grp,pdg,alpha)
    ops  = [self._accum_grad_v, self._loss0, self._loss1, self._loss2, self._losstotal]
    if summary:
      ops += [self._merged_summary]
    ops += [self._num_point,self._num_sim_grp0,self._num_sim_grp1,self._num_sim_grp2,self._elements]
    return sess.run(ops, feed_dict = feed_dict)

  def zero_gradients(self, sess):
    if not self._flags.TRAIN:
      raise NotImplementedError
    return sess.run([self._zero_grad])
    
  def apply_gradient(self,sess):
    if not self._flags.TRAIN:
      raise NotImplementedError
    return sess.run(self._apply_grad)
