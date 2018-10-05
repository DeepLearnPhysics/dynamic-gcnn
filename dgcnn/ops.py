from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as L
import tensorflow.contrib.slim as slim
def k_nn(points, k):
  # The shape of points (B, N, C)
  # Use linear algebra to find all pairwise distance in parallel
  M = points
  transpose  = tf.transpose(M, perm=[0, 2, 1])
  inner_prod = tf.matmul(M, transpose)
  squared    = tf.reduce_sum(tf.square(M), axis=-1, keep_dims=True)
  squared_tranpose = tf.transpose(squared, perm=[0, 2, 1])
  nn_dist = squared + squared_tranpose - 2 * inner_prod
  # Next pick the top k shortest ones
  _, idx = tf.nn.top_k(-nn_dist,k=k)
  return idx

def edges(points, k=20):
  # points (B, N, C)
  knn_idx = k_nn(points, k)

  points_central = points
  batch_size   = points.get_shape()[0].value
  num_points   = points.get_shape()[1].value
  num_dims     = points.get_shape()[2].value

  idx_ = tf.range   (batch_size) * num_points
  idx_ = tf.reshape (idx_, [batch_size, 1, 1])

  points_flat      = tf.reshape  (points, [-1, num_dims])
  points_neighbors = tf.gather   (points_flat, knn_idx+idx_)
  points_central   = tf.expand_dims (points_central, axis=-2)

  points_central = tf.tile (points_central, [1, 1, k, 1])

  edge_feature = tf.concat([points_central, points_neighbors-points_central], axis=-1)
  return edge_feature

def edge_conv(point_cloud, k, num_filters, trainable, debug=False):

  net = point_cloud
  net = edges(net, k=k)
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  net = slim.conv2d(inputs      = net,
                    num_outputs = num_filters,
                    kernel_size = 1,
                    stride      = 1,
                    trainable   = trainable,
                    padding     = 'VALID',
                    normalizer_fn = slim.batch_norm,
                    #activation_fn = None,
                    scope       = 'conv0')
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  net_max  = tf.reduce_max  (net, axis=-2, keep_dims=True)
  net_mean = tf.reduce_mean (net, axis=-2, keep_dims=True)
  net = tf.concat([net_max, net_mean], axis=-1)
  if debug: print('Shape {:s} ... Name {:s}'.format(net_max.shape.as_list(),net_max.name))
  if debug: print('Shape {:s} ... Name {:s}'.format(net_mean.shape.as_list(),net_mean.name))
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  net = slim.conv2d(inputs      = net,
                    num_outputs = 64,
                    kernel_size = 1,
                    stride      = 1,
                    trainable   = True,
                    padding     = 'VALID',
                    normalizer_fn = slim.batch_norm,
                    #activation_fn = None,
                    scope       = 'conv1')
  if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))
  
  return [net_max, net_mean, net]

def repeat_edge_conv(point_cloud, repeat, k, num_filters, trainable, debug=False):
  
  repeat = int(repeat)
  if not type(k) == type(list()):
    k = [int(k)] * repeat
  elif not len(k) == repeat:
    print('Length of k != repeat')
    raise ValueError
  if not type(num_filters) == type(list()):
    num_filters = [int(num_filters)] * repeat
  elif not len(num_filters) == repeat:
    print('Length of num_filters != repeat')
    raise ValueError
    
  net = point_cloud
  tensors = []
  for i in range(repeat):
    scope = 'EdgeConv%d' % i
    with tf.variable_scope(scope):
      tensors += edge_conv(net, k[i], num_filters[i], trainable, debug)
      net = tensors[-1]
      net = tf.squeeze(net,axis=-2)

  return tensors

def fc(net, repeat, num_filters, trainable, debug=False):

  repeat = int(repeat)
  if not type(num_filters) == type(list()):
    num_filters = [int(num_filters)] * repeat
  elif not len(num_filters) == repeat:
    print('Length of num_filters != repeat')
    raise ValueError

  for i in range(repeat):
    scope='FC%d' % i
    net = slim.conv2d(inputs      = net,
                      num_outputs = num_filters[i],
                      kernel_size = 1,
                      stride      = 1,
                      trainable   = trainable,
                      padding     = 'VALID',
                      normalizer_fn = slim.batch_norm,
                      #activation_fn = None,
                      scope       = scope)
    if debug: print('Shape {:s} ... Name {:s}'.format(net.shape.as_list(),net.name))

  return net
