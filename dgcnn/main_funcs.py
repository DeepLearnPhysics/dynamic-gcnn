from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,datetime,sys
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
  num_entries = io.num_entries()
  ctr = 0
  while ctr < num_entries:
    idx,data,label,weight=io.next()
    msg = str(ctr) + '/' + str(num_entries) + ' ... '  + str(idx) + ' ' + str(data[0].shape)
    if label:
      msg += str(label[0].shape)
    if weight:
      msg += str(weight[0].shape)
    print(msg)
    ctr += len(data)
  io.finalize()
  
class Handlers:
  sess         = None
  data_io      = None
  csv_logger   = None
  weight_io    = None
  train_logger = None
  iteration    = 0

def train(flags):

  flags.TRAIN = True
  handlers = prepare(flags)
  train_loop(flags,handlers)

def inference(flags):

  flags.TRAIN = False
  handlers = prepare(flags)
  inference_loop(flags,handlers)
  
def prepare(flags):

  handlers = Handlers()
  # assert
  if flags.BATCH_SIZE % (flags.MINIBATCH_SIZE * len(flags.GPUS)):
    msg = '--batch_size (%d) must be a modular of --gpus (%d) * --minibatch_size (%d)\n'
    msg = msg % (flags.BATCH_SIZE,flags.MINIBATCH_SIZE,len(flags.GPUS))
    sys.stderr.write(msg)
    sys.exit(1)

  # IO configuration
  handlers.data_io = dgcnn.io_factory(flags)
  handlers.data_io.initialize()
  _,train_data,_,_ = handlers.data_io.next()

  # Trainer configuration
  flags.NUM_CHANNEL = handlers.data_io.num_channels()
  handlers.trainer = dgcnn.trainval(flags)
  handlers.trainer.initialize()

  # Create a session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  config.allow_soft_placement = True
  handlers.sess = tf.Session(config=config)
  init = tf.group(tf.global_variables_initializer(),
                  tf.local_variables_initializer())
  handlers.sess.run(init)

  handlers.weight_io = tf.train.Saver(max_to_keep=flags.CHECKPOINT_NUM,
                                      keep_checkpoint_every_n_hours=flags.CHECKPOINT_HOUR)
  if flags.WEIGHT_PREFIX:
    save_dir = flags.WEIGHT_PREFIX[0:flags.WEIGHT_PREFIX.rfind('/')]
    if save_dir and not os.path.isdir(save_dir): os.makedirs(save_dir)

  handlers.iteration = 0
  loaded_iteration   = 0
  if flags.MODEL_PATH:
    handlers.weight_io.restore(handlers.sess, flags.MODEL_PATH)
    loaded_iteration = iteration_from_filename(flags.MODEL_PATH)
    if flags.TRAIN: handlers.iteration = loaded_iteration+1

  if flags.LOG_DIR:
    if not os.path.exists(flags.LOG_DIR): os.mkdir(flags.LOG_DIR)
    handlers.train_logger = tf.summary.FileWriter(flags.LOG_DIR)
    handlers.train_logger.add_graph(handlers.sess.graph)
    logname = '%s/train_log-%07d.csv' % (flags.LOG_DIR,loaded_iteration)
    if not flags.TRAIN:
      logname = '%s/inference_log-%07d.csv' % (flags.LOG_DIR,loaded_iteration)
    handlers.csv_logger = open(logname,'w')
    
  return handlers
    
def train_loop(flags,handlers):

  handlers.csv_logger.write('iter,epoch')
  handlers.csv_logger.write(',titer,ttrain,tio,tsave,tsummary')
  handlers.csv_logger.write(',tsumiter,tsumtrain,tsumio,tsumsave,tsumsummary')
  handlers.csv_logger.write(',loss,accuracy\n')
  
  tsum       = 0.
  tsum_train = 0.
  tsum_io    = 0.
  tsum_save  = 0.
  tsum_summary = 0.
  while handlers.iteration < flags.ITERATION:

    tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    tstart_iteration = time.time()
    
    report_step  = flags.REPORT_STEP and ((handlers.iteration+1) % flags.REPORT_STEP == 0)
    summary_step = flags.SUMMARY_STEP and handlers.train_logger and ((handlers.iteration+1) % flags.SUMMARY_STEP == 0)
    checkpt_step = flags.CHECKPOINT_STEP and flags.WEIGHT_PREFIX and ((handlers.iteration+1) % flags.CHECKPOINT_STEP == 0)

    tstart = time.time()
    idx,data,label,weight = handlers.data_io.next()
    tspent_io = time.time() - tstart
    tsum_io += tspent_io
    
    current_idx = 0
    loss_v = []
    accuracy_v  = []
    handlers.trainer.zero_gradients(handlers.sess)
    # Accummulate gradients
    tspent_train = 0.
    tspent_summary = 0.
    while current_idx < flags.BATCH_SIZE:
      tstart   = time.time()
      data_v   = []
      label_v  = []
      weight_v = None
      if weight is not None: weight_v = []
      for _ in flags.GPUS:
        start = current_idx
        end   = current_idx + flags.MINIBATCH_SIZE
        data_v.append(data[start:end])
        label_v.append(label[start:end])
        if weight is not None:
          weight_v.append(weight[start:end])
        current_idx = end
      # compute gradients
      make_summary = summary_step and (current_idx == flags.BATCH_SIZE)
      res = handlers.trainer.accum_gradient(handlers.sess,data_v,label_v,weight_v,summary=make_summary)
      accuracy_v.append(res[1])
      loss_v.append(res[2])
      tspent_train = tspent_train + (time.time() - tstart)
      # log summary
      if make_summary:
        tstart = time.time()
        handlers.train_logger.add_summary(res[3],handlers.iteration)
        tspent_summary = time.time() - tstart
    # Apply gradients
    tstart = time.time()
    handlers.trainer.apply_gradient(handlers.sess)
    tspent_train = tspent_train + (time.time() - tstart)

    tsum_train += tspent_train
    tsum_summary += tspent_summary
    
    # Compute loss/accuracy
    loss = np.mean(loss_v)
    accuracy = np.mean(accuracy_v)
    epoch = handlers.iteration * float(flags.BATCH_SIZE) / handlers.data_io.num_entries()
    # Save snapshot
    tspent_save = 0.
    if checkpt_step:
      tstart = time.time()
      ssf_path = handlers.weight_io.save(handlers.sess,flags.WEIGHT_PREFIX,global_step=handlers.iteration)
      tspent_save = time.time() - tstart
      print('saved @',ssf_path)
    # Report (logger)
    if handlers.csv_logger:
      tspent_iteration = time.time() - tstart_iteration
      tsum += tspent_iteration
      csv_data  = '%d,%g,' % (handlers.iteration,epoch)
      csv_data += '%g,%g,%g,%g,%g,' % (tspent_iteration,tspent_train,tspent_io,tspent_save,tspent_summary)
      csv_data += '%g,%g,%g,%g,%g,' % (tsum,tsum_train,tsum_io,tsum_save,tsum_summary)
      csv_data += '%g,%g\n' % (loss,accuracy)
      handlers.csv_logger.write(csv_data)
    # Report (stdout)
    if report_step:
      loss = round_decimals(loss,4)
      accuracy = round_decimals(accuracy,4)
      tfrac = round_decimals(tspent_train/tspent_iteration*100.,2)
      epoch = round_decimals(epoch,2)
      mem = handlers.sess.run(tf.contrib.memory_stats.MaxBytesInUse())
      msg = 'Iteration %d (epoch %g) @ %s ... train time fraction %g%% max mem. %g ... loss %g accuracy %g'
      msg = msg % (handlers.iteration,epoch,tstamp_iteration,tfrac,mem,loss,accuracy)
      print(msg)
      sys.stdout.flush()
      if handlers.csv_logger: handlers.csv_logger.flush()
      if handlers.train_logger: handlers.train_logger.flush()
    # Increment iteration counter
    handlers.iteration +=1

  handlers.train_logger.close()
  handlers.csv_logger.close()
  handlers.data_io.finalize()

def inference_loop(flags,handlers):
  handlers.csv_logger.write('iter,epoch')
  handlers.csv_logger.write(',titer,tinference,tio')
  handlers.csv_logger.write(',tsumiter,tsuminference,tsumio')
  handlers.csv_logger.write(',loss,accuracy\n')
  tsum           = 0.
  tsum_io        = 0.
  tsum_inference = 0.
  while handlers.iteration < flags.ITERATION:

    tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    tstart_iteration = time.time()
    
    report_step  = flags.REPORT_STEP and ((handlers.iteration+1) % flags.REPORT_STEP == 0)

    tstart = time.time()
    idx,data,label,weight = handlers.data_io.next()
    tspent_io = time.time() - tstart
    tsum_io += tspent_io
    
    current_idx = 0
    softmax_vv = []
    loss_v = []
    accuracy_v  = []
    
    # Run inference
    tspent_inference = 0.
    tstart = time.time()
    while current_idx < flags.BATCH_SIZE:
      data_v   = []
      label_v  = None
      weight_v = None
      if label  is not None: label_v  = []
      if weight is not None: weight_v = []
      for _ in flags.GPUS:
        start = current_idx
        end   = current_idx + flags.MINIBATCH_SIZE
        data_v.append(data[start:end])
        if label  is not None:
          label_v.append(label[start:end])
        if weight is not None:
          weight_v.append(weight[start:end])
        current_idx = end
      # compute gradients
      res = handlers.trainer.inference(handlers.sess,data_v,label_v,weight_v)
      if flags.LABEL_KEY:
        softmax_vv = softmax_vv + res[0:-2]
        accuracy_v.append(res[-2])
        loss_v.append(res[-1])
      else:
        softmax_vv = softmax_vv + res
    tspent_inference = tspent_inference + (time.time() - tstart)
    tsum_inference  += tspent_inference

    # Store output if requested
    if flags.OUTPUT_FILE:
      idx_ctr = 0
      for softmax_v in softmax_vv:
        for softmax in softmax_v:
          handlers.data_io.store(idx[idx_ctr],softmax)
          idx_ctr += 1
    
    # Compute loss/accuracy
    loss,accuracy=[-1,-1]
    if flags.LABEL_KEY:
      loss = np.mean(loss_v)
      accuracy = np.mean(accuracy_v)
    epoch = handlers.iteration * float(flags.BATCH_SIZE) / handlers.data_io.num_entries()
    # Report (logger)
    if handlers.csv_logger:
      tspent_iteration = time.time() - tstart_iteration
      tsum += tspent_iteration
      csv_data  = '%d,%g,' % (handlers.iteration,epoch)
      csv_data += '%g,%g,%g,' % (tspent_iteration,tspent_inference,tspent_io)
      csv_data += '%g,%g,%g,' % (tsum,tsum_inference,tsum_io)
      csv_data += '%g,%g\n' % (loss,accuracy)
      handlers.csv_logger.write(csv_data)
    # Report (stdout)
    if report_step:
      loss = round_decimals(loss,4)
      accuracy = round_decimals(accuracy,4)
      tfrac = round_decimals(tspent_inference/tspent_iteration*100.,2)
      epoch = round_decimals(epoch,2)
      mem = handlers.sess.run(tf.contrib.memory_stats.MaxBytesInUse())
      msg = 'Iteration %d (epoch %g) @ %s ... inference time fraction %g%% max mem. %g ... loss %g accuracy %g'
      msg = msg % (handlers.iteration,epoch,tstamp_iteration,tfrac,mem,loss,accuracy)
      print(msg)
      sys.stdout.flush()
      if handlers.csv_logger: handlers.csv_logger.flush()
    # Increment iteration counter
    handlers.iteration +=1

  handlers.csv_logger.close()
  handlers.data_io.finalize()

