from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.python.platform
import tensorflow as tf
import tensorflow.contrib.slim as slim
import dgcnn

def build(point_cloud, flags):

  num_edge_conv = int(flags.NUM_EDGE_CONV)
  is_training   = bool(flags.TRAIN)
  k = int(flags.KVALUE)
  debug = bool(flags.DEBUG)
  num_class = int(flags.NUM_CLASS)
  
  net = point_cloud
  batch_size = net.get_shape()[0].value
  num_point  = net.get_shape()[1].value
  if debug:
    print('\n')
    print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))

  tensors = dgcnn.ops.repeat_edge_conv(net, repeat=num_edge_conv, k=k, num_filters=64, trainable=is_training, debug=debug)
  
  concat = []
  for i in range(num_edge_conv):
    concat.append(tensors[3*i+2])
  concat = tf.concat(concat,axis=-1)

  net = slim.conv2d(inputs      = concat,
                    num_outputs = 1024,
                    kernel_size = 1,
                    stride      = 1,
                    trainable   = True,
                    padding     = 'VALID',
                    normalizer_fn = slim.batch_norm,
                    #activation_fn = None,
                    scope       = 'MergedEdgeConv')
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  tensors.append(net)

  net = slim.max_pool2d(net, kernel_size=[num_point,1], scope='maxpool0')
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  net  = tf.tile(net, [1, num_point, 1, 1])
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  concat = [net] + tensors

  net = tf.concat(values=concat, axis=3)
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))

  net = dgcnn.ops.fc(net=net, repeat=2, num_filters=[512,256], trainable=is_training, debug=debug)

  if is_training:
    net = tf.nn.dropout(net, 0.7, None)
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))

  net = slim.conv2d(inputs      = net,
                    num_outputs = num_class,
                    kernel_size = 1,
                    stride      = 1,
                    trainable   = True,
                    padding     = 'VALID',
                    normalizer_fn = slim.batch_norm,
                    #activation_fn = None,
                    scope       = 'Final')
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  
  net = tf.squeeze(net, [2])
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  return net

