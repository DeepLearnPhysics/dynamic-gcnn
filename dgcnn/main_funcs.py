from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,datetime
import numpy as np
import dgcnn
import tensorflow as tf

def round_decimals(val,digits):
  factor = float(np.power(10,digits))
  return int(val * factor+0.5) / factor

def iteration_from_filename(file_name):
  return int((file_name.split('-'))[-1])

def iotest(flags):
  # IO configuration
  io = dgcnn.io_factory(flags)
  io.initialize()
  for i in range(10):
    data,label,_ = io.next()
    #print(i,np.shape(data),np.shape(label))
    print(i,data[0].shape,label[0].shape)

def train(flags):

  # assert
  if flags.BATCH_SIZE % (flags.MINIBATCH_SIZE * len(flags.GPUS)):
    sys.stderr.write('--batch_size must be a modular of --gpus * --minibatch_size\n')
    sys.exit(1)

  # IO configuration
  io = dgcnn.io_factory(flags)
  io.initialize()
  train_data,_,_ = io.next()

  # Trainer configuration
  flags.NUM_CHANNEL = train_data[0].shape[-1]
  trainer = dgcnn.trainval(flags)
  trainer.initialize()

  # Create a session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  sess = tf.Session(config=config)
  init = tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer())
  sess.run(init)

  train_writer, csv_logger = (None,None)
  if flags.LOG_DIR:
    if not os.path.exists(flags.LOG_DIR): os.mkdir(flags.LOG_DIR)
    train_writer = tf.summary.FileWriter(flags.LOG_DIR)
    train_writer.add_graph(sess.graph)
    csv_logger = open('%s/log.csv' % flags.LOG_DIR,'w')
    csv_logger.write('iter,epoch,titer,ttrain,tio,tsave,tsummary,tsumiter,tsumtrain,tsumio,tsumsave,tsumsummary,loss,accuracy\n')
    
  saver = tf.train.Saver(max_to_keep=flags.CHECKPOINT_NUM,
                         keep_checkpoint_every_n_hours=flags.CHECKPOINT_HOUR)
  if flags.WEIGHT_PREFIX:
    save_dir = flags.WEIGHT_PREFIX[0:flags.WEIGHT_PREFIX.rfind('/')]
    if save_dir and not os.path.isdir(save_dir): os.makedirs(save_dir)

  iteration = 0
  if flags.MODEL_PATH:
    saver.restore(sess, flags.MODEL_PATH)
    iteration = iteration_from_filename(flags.MODEL_PATH)+1

  tsum       = 0.
  tsum_train = 0.
  tsum_io    = 0.
  tsum_save  = 0.
  tsum_summary = 0.
  while iteration < flags.ITERATION:

    tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    tstart_iteration = time.time()
    
    report_step  = flags.REPORT_STEP and ((iteration+1) % flags.REPORT_STEP == 0)
    summary_step = flags.SUMMARY_STEP and train_writer and ((iteration+1) % flags.SUMMARY_STEP == 0)
    checkpt_step = flags.CHECKPOINT_STEP and flags.WEIGHT_PREFIX and ((iteration+1) % flags.CHECKPOINT_STEP == 0)

    tstart = time.time()
    data,label,idx = io.next()
    tspent_io = time.time() - tstart
    tsum_io += tspent_io
    
    current_idx = 0
    loss_v = []
    accuracy_v  = []
    trainer.zero_gradients(sess)
    # Accummulate gradients
    tspent_train = 0.
    tspent_summary = 0.
    while current_idx < flags.BATCH_SIZE:
      tstart = time.time()
      data_v  = []
      label_v = []
      for _ in flags.GPUS:
        start = current_idx
        end   = current_idx + flags.MINIBATCH_SIZE
        data_v.append(data[start:end])
        label_v.append(label[start:end])
        current_idx = end
      # compute gradients
      make_summary = summary_step and (current_idx == flags.BATCH_SIZE)
      res = trainer.accum_gradient(sess,data_v,label_v,summary=make_summary)
      accuracy_v.append(res[1])
      loss_v.append(res[2])
      tspent_train = tspent_train + (time.time() - tstart)
      # log summary
      if make_summary:
        tstart = time.time()
        train_writer.add_summary(res[3],iteration)
        tspent_summary = time.time() - tstart
    # Apply gradients
    tstart = time.time()
    trainer.apply_gradient(sess)
    tspent_train = tspent_train + (time.time() - tstart)

    tsum_train += tspent_train
    tsum_summary += tspent_summary
    
    # Compute loss/accuracy
    loss = np.mean(loss_v)
    accuracy = np.mean(accuracy_v)
    epoch = iteration * float(flags.BATCH_SIZE) / io.num_entries()
    # Save snapshot
    tspent_save = 0.
    if checkpt_step:
      tstart = time.time()
      ssf_path = saver.save(sess,flags.WEIGHT_PREFIX,global_step=iteration)
      tspent_save = time.time() - tstart
      print('saved @',ssf_path)
    # Log csv
    if csv_logger:
      tspent_iteration = time.time() - tstart_iteration
      tsum += tspent_iteration
      csv_data  = '%d,%g,' % (iteration,epoch)
      csv_data += '%g,%g,%g,%g,%g,' % (tspent_iteration,tspent_train,tspent_io,tspent_save,tspent_summary)
      csv_data += '%g,%g,%g,%g,%g,' % (tsum,tsum_train,tsum_io,tsum_save,tsum_summary)
      csv_data += '%g,%g\n' % (loss,accuracy)
      csv_logger.write(csv_data)
    # Report (stdout)
    if report_step:
      loss = round_decimals(loss,4)
      accuracy = round_decimals(accuracy,4)
      tfrac = round_decimals(tspent_train/tspent_iteration*100.,2)
      epoch = round_decimals(epoch,2)
      mem = sess.run(tf.contrib.memory_stats.MaxBytesInUse())
      msg = 'Iteration %d (epoch %g) @ %s ... train time fraction %g%% max mem. %g ... loss %g accuracy %g'
      msg = msg % (iteration,epoch,tstamp_iteration,tfrac,mem,loss,accuracy)
      print(msg)
      if csv_logger: csv_logger.flush()
      if train_writer: train_writer.flush()
    # Increment iteration counter
    iteration +=1

  train_writer.close()
  csv_logger.close()
  io.finalize()
