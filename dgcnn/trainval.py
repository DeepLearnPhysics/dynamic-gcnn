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

      self._points_v = []
      self._labels_v = []
      loss_v     = []
      accuracy_v = []
      grad_v     = []

      for i, gpu_id in enumerate(self._flags.GPUS):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('GPU%d' % gpu_id) as scope:
            with tf.variable_scope("dgcnn",reuse=tf.AUTO_REUSE):
              points = tf.placeholder(tf.float32, 
                                      shape=(self._flags.MINIBATCH_SIZE,self._flags.NUM_POINT,self._flags.NUM_CHANNEL))
              labels = tf.placeholder(tf.int32,
                                      shape=(self._flags.MINIBATCH_SIZE,self._flags.NUM_POINT))
              self._points_v.append(points)
              self._labels_v.append(labels)
              pred = dgcnn.model.build(points, self._flags)
              loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels)
              loss = tf.reduce_mean(loss)
              loss_v.append(loss)
              correct  = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels))
              accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
              accuracy_v.append(accuracy)
              # If training, compute gradients
              if self._flags.TRAIN:
                grad = self._optimizer.compute_gradients(loss)
                grad_v.append(grad)
            
      # Average loss & accuracy across GPUs
      self._loss     = tf.add_n(loss_v) / float(len(self._flags.GPUS))
      self._accuracy = tf.add_n(accuracy_v) / float(len(self._flags.GPUS))
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
        tf.summary.scalar('accuracy', self._accuracy)
        tf.summary.scalar('loss', self._loss)
        self._merged_summary=tf.summary.merge_all()
      
  def feed_dict(self,data,label=None):
    res = {}
    for i,gpu_id in enumerate(self._flags.GPUS):
      res[self._points_v [i]] = data  [i]
      if label is not None:
        res[self._labels_v [i]] = label [i]
    return res

  def make_summary(self, sess, data, label):
    if not self._flags.TRAIN:
      raise NotImplementedError    
    feed_dict = self.feed_dict(data,label)
    return sess.run(self._merged_summary,feed_dict=feed_dict)
  
  def inference(self,data,label=None):
    feed_dict = self.feed_dict(data,label)
    ops  = [self._accum_grad_v]
    ops += [self._accuracy]
    if label is not None:
      ops += [self._loss]
    return sess.run(ops, feed_dict = feed_dict)    
    
  def accum_gradient(self, sess, data, label, summary=False):
    if not self._flags.TRAIN:
      raise NotImplementedError
    
    feed_dict = self.feed_dict(data,label)
    ops  = [self._accum_grad_v]
    ops += [self._accuracy, self._loss]
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
