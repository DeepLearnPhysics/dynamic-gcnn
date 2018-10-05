import os
import numpy as np

def train(flags):

  # assert
  if flags.BATCH_SIZE % (flags.MINIBATCH_SIZE * len(flags.GPUS)):
    sys.stderr.write('--batch_size must be a modular of --gpus * --minibatch_size\n')
    sys.exit(1)

  # IO configuration
  import iotool
  io = iotool.io_factory(flags)
  io.initialize()
  train_data,_,_ = io.next()

  # Trainer configuration
  from trainval import trainval
  flags.NUM_CHANNEL = train_data.shape[-1]
  trainer = trainval(flags)
  trainer.initialize()

  import tensorflow as tf
  
  if not os.path.exists(flags.LOG_DIR): os.mkdir(flags.LOG_DIR)
  
  # Create a session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess = tf.Session(config=config)
  init = tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer())
  sess.run(init)

  train_writer = tf.summary.FileWriter(os.path.join(flags.LOG_DIR, 'train'),
                                     sess.graph)
  test_writer = tf.summary.FileWriter(os.path.join(flags.LOG_DIR, 'test'))

  if flags.MODEL_PATH:
    saver.restore(sess, flags.MODEL_PATH)

  iteration = 0
  while iteration < flags.ITERATION:

    data,label,idx = io.next()
    data_v  = []
    label_v = []
    
    current_idx = 0
    loss_v = []
    accuracy_v  = []
    while current_idx < flags.BATCH_SIZE:
      for _ in flags.GPUS:
        start = current_idx
        end   = current_idx + flags.MINIBATCH_SIZE
        data_v.append(data[start:end])
        label_v.append(label[start:end])
        current_idx = end
      _,loss,accuracy = trainer.accum_gradient(sess,data_v,label_v)
      loss_v.append(loss)
      accuracy_v.append(accuracy)
    loss = np.array(loss_v).mean()
    accuracy = np.array(accuracy_v).mean()
    if (iteration+1) % flags.REPORT_STEP == 0:
      epoch = iteration * float(flags.BATCH_SIZE) / io.num_entries()
      print('Iteration %d (epoch %g) ... loss %g accuracy %g' % (iteration,epoch,loss,accuracy))
    trainer.apply_gradient(sess)

    iteration +=1

  
  
