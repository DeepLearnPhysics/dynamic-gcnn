from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,datetime,sys
import numpy as np
import dgcnn
import tensorflow as tf

#grouping = dgcnn.InclusiveGrouping
grouping = dgcnn.ScoreGrouping

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
  data_key = flags.DATA_KEYS[0]
  while ctr < num_entries:
    idx,blob=io.next()
    msg = str(ctr) + '/' + str(num_entries) + ' ... '  + str(idx) + ' ' + str(blob[data_key][0].shape)
    for key in flags.DATA_KEYS:
      if key == data_key: continue
      msg += str(blob[key][0].shape)
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
  _,blob = handlers.data_io.next()

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
    handlers.csv_logger = dgcnn.CSVData(logname)
    
  return handlers
    
def train_loop(flags,handlers):

  data_key, grp_key, pdg_key = flags.DATA_KEYS[0:3]
  tsum       = 0.
  tsum_train = 0.
  tsum_io    = 0.
  tsum_save  = 0.
  tsum_summary = 0.
  while handlers.iteration < flags.ITERATION:

    epoch = handlers.iteration * float(flags.BATCH_SIZE) / handlers.data_io.num_entries()
    tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    tstart_iteration = time.time()
    
    report_step  = flags.REPORT_STEP and ((handlers.iteration+1) % flags.REPORT_STEP == 0)
    summary_step = flags.SUMMARY_STEP and handlers.train_logger and ((handlers.iteration+1) % flags.SUMMARY_STEP == 0)
    checkpt_step = flags.CHECKPOINT_STEP and flags.WEIGHT_PREFIX and ((handlers.iteration+1) % flags.CHECKPOINT_STEP == 0)

    tstart = time.time()
    idx,blob = handlers.data_io.next()
    tspent_io = time.time() - tstart
    tsum_io += tspent_io

    data = blob[data_key]
    grp  = blob[grp_key]
    pdg  = blob[pdg_key]
    current_idx = 0
    losses = {'loss_same_group' : [],
              'loss_same_pdg'   : [],
              'loss_diff_group' : [],
              'loss_cluster'    : [],
              'loss_conf'       : [],
              'loss_total'      : [] }
    handlers.trainer.zero_gradients(handlers.sess)
    # Accummulate gradients
    tspent_train = 0.
    tspent_summary = 0.
    while current_idx < flags.BATCH_SIZE:
      tstart = time.time()
      data_v = []
      grp_v  = []
      pdg_v  = []
      for _ in flags.GPUS:
        start = current_idx
        end   = current_idx + flags.MINIBATCH_SIZE
        data_v.append(data[start:end])
        grp_v.append(grp[start:end])
        pdg_v.append(pdg[start:end])
        current_idx = end
      # compute gradients
      make_summary = summary_step and (current_idx == flags.BATCH_SIZE)
      alpha = min(flags.ALPHA_LIMIT, 1 + (float(epoch) / flags.ALPHA_DECAY) * (flags.ALPHA_LIMIT-1.) + 1.)
      res = handlers.trainer.accum_gradient(handlers.sess,data_v,grp_v,pdg_v,alpha,summary=make_summary)
      losses[ 'loss_same_group' ].append(res[1])
      losses[ 'loss_same_pdg'   ].append(res[2])
      losses[ 'loss_diff_group' ].append(res[3])
      losses[ 'loss_cluster'    ].append(res[4])
      losses[ 'loss_conf'       ].append(res[5])
      losses[ 'loss_total'      ].append(res[6])
      tspent_train = tspent_train + (time.time() - tstart)
      # log summary
      if make_summary:
        tstart = time.time()
        handlers.train_logger.add_summary(res[7],handlers.iteration)
        tspent_summary = time.time() - tstart
    # Apply gradients
    tstart = time.time()
    handlers.trainer.apply_gradient(handlers.sess)
    tspent_train = tspent_train + (time.time() - tstart)

    tsum_train += tspent_train
    tsum_summary += tspent_summary
    
    # Compute loss
    
    loss_same_group = np.mean(losses['loss_same_group'])
    loss_same_pdg   = np.mean(losses['loss_same_pdg'  ])
    loss_diff_group = np.mean(losses['loss_diff_group'])
    loss_cluster    = np.mean(losses['loss_cluster'   ])
    loss_conf       = np.mean(losses['loss_conf'      ])
    loss_total      = np.mean(losses['loss_total'     ])
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
      handlers.csv_logger.record(('iter','epoch'),(handlers.iteration,epoch))
      handlers.csv_logger.record(('titer','ttrain','tio','tsave','tsummary'),
                                 (tspent_iteration,tspent_train,tspent_io,tspent_save,tspent_summary))
      handlers.csv_logger.record(('tsumiter','tsumtrain','tsumio','tsumsave','tsumsummary'),
                                 (tsum,tsum_train,tsum_io,tsum_save,tsum_summary))
      handlers.csv_logger.record(('alpha','loss_same_group','loss_same_pdg','loss_diff_group','loss_cluster'),
                                 (alpha,loss_same_group,loss_same_pdg,loss_diff_group,loss_cluster))
      handlers.csv_logger.record(('loss_conf','loss_total'),(loss_conf,loss_total))
      handlers.csv_logger.write()
    # Report (stdout)
    if report_step:
      loss_same_group = round_decimals(loss_same_group, 4)
      loss_same_pdg   = round_decimals(loss_same_pdg,   4)
      loss_diff_group = round_decimals(loss_diff_group, 4)
      loss_cluster    = round_decimals(loss_cluster,    4)
      loss_conf       = round_decimals(loss_conf,       4)
      loss_total      = round_decimals(loss_total,      4)
      tfrac = round_decimals(tspent_train/tspent_iteration*100.,2)
      epoch = round_decimals(epoch,2)
      mem = round_decimals(handlers.sess.run(tf.contrib.memory_stats.MaxBytesInUse())/1.e9,3)
      msg1 = 'Iter. %d (epoch %g) @ %s ... ttrain %g%% mem. %g GB... alpha %g\n'
      msg1 = msg1 % (handlers.iteration,epoch,tstamp_iteration,tfrac,mem,alpha)
      msg2 = '  Cluster loss %g (%g+%g+%g), Score loss %g, Total Loss %g'
      msg2 = msg2 % (loss_cluster,loss_same_group,loss_same_pdg,loss_diff_group,loss_conf,loss_total)
      print(msg1,msg2)
      sys.stdout.flush()
      if handlers.csv_logger: handlers.csv_logger.flush()
      if handlers.train_logger: handlers.train_logger.flush()
    # Increment iteration counter
    handlers.iteration +=1

  handlers.train_logger.close()
  handlers.csv_logger.close()
  handlers.data_io.finalize()

def inference_loop(flags,handlers):

  data_key = flags.DATA_KEYS[0]
  grp_key,pdg_key=(None,None)
  if len(flags.DATA_KEYS)>1:
    grp_key = flags.DATA_KEYS[1]
  if len(flags.DATA_KEYS)>2:
    pdg_key = flags.DATA_KEYS[2]
  tsum           = 0.
  tsum_io        = 0.
  tsum_inference = 0.
  while handlers.iteration < flags.ITERATION:

    tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
    tstart_iteration = time.time()
    
    report_step  = flags.REPORT_STEP and ((handlers.iteration+1) % flags.REPORT_STEP == 0)

    tstart = time.time()
    idx,blob = handlers.data_io.next()
    tspent_io = time.time() - tstart
    tsum_io += tspent_io

    data = blob[data_key]
    grp,pdg=(None,None)
    if grp_key is not None: grp = blob[grp_key]
    if pdg_key is not None: pdg = blob[pdg_key]
    losses = {'loss_same_group' : [],
              'loss_same_pdg'   : [],
              'loss_diff_group' : [],
              'loss_cluster'    : [],
              'loss_conf'       : [],
              'loss_total'      : [] }
    current_idx = 0
    
    # Run inference
    tspent_inference = 0.
    tstart   = time.time()
    pred_vv  = []
    score_vv = []
    while current_idx < flags.BATCH_SIZE:
      data_v = []
      grp_v,pdg_v=(None,None)
      if grp_key is not None:
        grp_v,pdg_v=[],[]
      for _ in flags.GPUS:
        start = current_idx
        end   = current_idx + flags.MINIBATCH_SIZE
        data_v.append(data[start:end])
        if grp_v is not None: grp_v.append(grp[start:end])
        if pdg_v is not None: pdg_v.append(pdg[start:end])
        current_idx = end
      # compute gradients
      alpha = flags.ALPHA_LIMIT
      res = handlers.trainer.inference(handlers.sess,data_v,grp_v,pdg_v,alpha)
      num_gpus = len(flags.GPUS)
      pred_vv.append(res[0:num_gpus])
      score_vv.append(res[num_gpus:num_gpus*2])
      losses[ 'loss_same_group' ].append(res[num_gpus*2])
      losses[ 'loss_same_pdg'   ].append(res[num_gpus*2+1])
      losses[ 'loss_diff_group' ].append(res[num_gpus*2+2])
      losses[ 'loss_cluster'    ].append(res[num_gpus*2+3])
      losses[ 'loss_conf'       ].append(res[num_gpus*2+4])
      losses[ 'loss_total'      ].append(res[num_gpus*2+5])

    tspent_inference = tspent_inference + (time.time() - tstart)
    tsum_inference  += tspent_inference
    # Compute loss
    loss_same_group = np.mean(losses['loss_same_group'])
    loss_same_pdg   = np.mean(losses['loss_same_pdg'  ])
    loss_diff_group = np.mean(losses['loss_diff_group'])
    loss_cluster    = np.mean(losses['loss_cluster'   ])
    loss_conf       = np.mean(losses['loss_conf'      ])
    loss_total      = np.mean(losses['loss_total'     ])

    # Store output if requested
    if flags.OUTPUT_FILE:
      idx_ctr = 0
      for i in range(len(pred_vv)):
        for j in range(len(pred_vv[i])):
          pred  = pred_vv[i][j]
          score = score_vv[i][j]
          print('score min,max,mean,std',score.min(),score.max(),score.mean(),score.std())
          grp = grouping(pred,score,5)
          handlers.data_io.store(idx[idx_ctr],grp[0].reshape([-1,1]))
          idx_ctr += 1

    epoch = handlers.iteration * float(flags.BATCH_SIZE) / handlers.data_io.num_entries()
    # Report (logger)
    if handlers.csv_logger:
      tspent_iteration = time.time() - tstart_iteration
      tsum += tspent_iteration

      handlers.csv_logger.record(('iter','epoch'),(handlers.iteration,epoch))
      handlers.csv_logger.record(('titer','tinference','tio'),
                                 (tspent_iteration,tspent_inference,tspent_io))
      handlers.csv_logger.record(('tsumiter','tsuminference','tsumio'),
                                 (tsum,tsum_inference,tsum_io))
      handlers.csv_logger.record(('alpha','loss_same_group','loss_same_pdg','loss_diff_group','loss_cluster'),
                                 (alpha,loss_same_group,loss_same_pdg,loss_diff_group,loss_cluster))
      handlers.csv_logger.record(('loss_conf','loss_total'),(loss_conf,loss_total))
      handlers.csv_logger.write()
    # Report (stdout)
    if report_step:
      loss_same_group = round_decimals(loss_same_group, 4)
      loss_same_pdg   = round_decimals(loss_same_pdg,   4)
      loss_diff_group = round_decimals(loss_diff_group, 4)
      loss_cluster    = round_decimals(loss_cluster,    4)
      loss_conf       = round_decimals(loss_conf,       4)
      loss_total      = round_decimals(loss_total,      4)
      tfrac = round_decimals(tspent_inference/tspent_iteration*100.,2)
      epoch = round_decimals(epoch,2)
      mem = round_decimals(handlers.sess.run(tf.contrib.memory_stats.MaxBytesInUse())/1.e9,3)
      msg1 = 'Iter. %d (epoch %g) @ %s ... ttrain %g%% mem. %g GB... alpha %g\n'
      msg1 = msg1 % (handlers.iteration,epoch,tstamp_iteration,tfrac,mem,alpha)
      msg2 = '  Cluster loss %g (%g+%g+%g), Score loss %g, Total Loss %g'
      msg2 = msg2 % (loss_cluster,loss_same_group,loss_same_pdg,loss_diff_group,loss_conf,loss_total)
      print(msg1,msg2)
      sys.stdout.flush()
      if handlers.csv_logger: handlers.csv_logger.flush()
    # Increment iteration counter
    handlers.iteration +=1

  handlers.csv_logger.close()
  handlers.data_io.finalize()

