import tensorflow.python.platform
import tensorflow as tf
import tensorflow.contrib.slim as slim
import ops

def build(point_cloud, flags):

  num_edge_conv = int(flags.NUM_EDGE_CONV)
  is_training   = bool(flags.TRAIN)
  k = int(flags.KVALUE)
  debug = bool(flags.DEBUG)
  
  net = point_cloud
  batch_size = net.get_shape()[0].value
  num_point  = net.get_shape()[1].value
  print('input',net.shape)

  tensors = ops.repeat_edge_conv(net, repeat=num_edge_conv, k=k, num_filters=64, trainable=is_training, debug=True)
  
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
  if debug: print('Net',net.shape)
  tensors.append(net)

  net = slim.max_pool2d(net, kernel_size=[num_point,1], scope='maxpool0')
  if debug: print('MaxPool',net.shape)
  net  = tf.tile(net, [1, num_point, 1, 1])
  if debug: print('Tile',net.shape)
  concat = [net] + tensors

  net = tf.concat(values=concat, axis=3)
  if debug: print('Net',net.shape)

  net = ops.fc(net=net, repeat=2, num_filters=[512,256], trainable=is_training, debug=True)

  if is_training:
    net = tf.nn.dropout(net, 0.7, None)
    if debug: print('Dropout',net.shape)

  net = slim.conv2d(inputs      = net,
                    num_outputs = 2,
                    kernel_size = 1,
                    stride      = 1,
                    trainable   = True,
                    padding     = 'VALID',
                    normalizer_fn = slim.batch_norm,
                    #activation_fn = None,
                    scope       = 'Final')
  if debug: print('Final',net.shape)
  
  net = tf.squeeze(net, [2])
  if debug: print('Squeeze',net.shape)
  return net

