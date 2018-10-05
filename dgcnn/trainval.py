import model
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
        with tf.device('/gpu:%d' % gpu_id):
          with tf.variable_scope("dgcnn",reuse=tf.AUTO_REUSE):
            points = tf.placeholder(tf.float32, 
                                    shape=(self._flags.MINIBATCH_SIZE,self._flags.NUM_POINT,self._flags.NUM_CHANNEL))
            labels = tf.placeholder(tf.int32,
                                    shape=(self._flags.MINIBATCH_SIZE,self._flags.NUM_POINT))
            self._points_v.append(points)
            self._labels_v.append(labels)
            
            pred = model.build(points, self._flags)
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=labels)
            loss = tf.reduce_mean(loss) #/ float(self._flags.MINIBATCH_SIZE * self._flags.NUM_POINT)
            print(loss.shape)
            correct  = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels))
            #accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(self._flags.MINIBATCH_SIZE * self._flags.NUM_POINT)
            accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
            grad = self._optimizer.compute_gradients(loss)
            grad_v.append(grad)
            accuracy_v.append(accuracy)
            loss_v.append(loss)

      # Synchronization
      self._loss     = tf.add_n(loss_v) / float(len(self._flags.GPUS))
      print(self._loss.shape)
      self._accuracy = tf.add_n(accuracy_v) / float(len(self._flags.GPUS))
      accum_vars   = [tf.Variable(v.initialized_value(),trainable=False) for v in tf.trainable_variables()]
      self._zero_grad    = [v.assign(tf.zeros_like(v)) for v in accum_vars]
      self._accum_grad_v = []

      # per-GPU operation
      for i,gpu_id in enumerate(self._flags.GPUS):
        grad = grad_v[i]
        self._accum_grad_v += [accum_vars[j].assign_add(g[0]) for j,g in enumerate(grad)]

      self._apply_grad = self._optimizer.apply_gradients(zip(accum_vars, tf.trainable_variables()))
    

  def feed_dict(self,data,label):

    res = {}
    for i,gpu_id in enumerate(self._flags.GPUS):
      res[self._points_v [i]] = data  [i]
      res[self._labels_v [i]] = label [i]

    return res
    
  def accum_gradient(self, sess, data, label):

    feed_dict = self.feed_dict(data,label)

    ops  = [self._accum_grad_v]
    ops += [self._loss, self._accuracy]

    return sess.run(ops, feed_dict = feed_dict)

  def apply_gradient(self,sess):

    return sess.run(self._apply_grad)

  
