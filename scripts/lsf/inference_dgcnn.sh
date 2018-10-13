#!/bin/bash

MODEL_NAME=$1; # [dgcnn,residual-dgcnn,residual-dgcnn-nofc]
NUM_LAYERS=$2; # recommended: 6
ITERATION=$3;  # 
WEIGHT_ID=$4;  
LEARNING_RATE=$5;
KVALUE=40;
NUM_FILTERS=64;
WEIGHT_KEY=""; # leaving it empty means no weight
GPUS=`python -c "import os;print len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))"`;

IN_STORAGE_DIR=/gpfs/slac/staas/fs1/g/neutrino/kterao;
OUT_STORAGE_DIR=/gpfs/slac/staas/fs1/g/neutrino/${USER};
SWDIR=$HOME/sw;
DGCNN_DIR=$SWDIR/dynamic-gcnn;
WEIGHT_FILE=/gpfs/slac/staas/fs1/g/neutrino/kterao/${MODEL_NAME}/res${NUM_LAYERS}/weights/snapshot-${WEIGHT_ID}
TEST_FILE=dlprod_ppn_v08_p02_test.root;
WORK_DIR=/scratch/kterao/temp_$$;
DATA_DIR=/scratch/kterao/data_$$;


source $SWDIR/larcv2/configure.sh;
mkdir -p $WORK_DIR;
mkdir -p $DATA_DIR;
cp $IN_STORAGE_DIR/data/$TEST_FILE $DATA_DIR;
cd $WORK_DIR;
echo $CUDA_VISIBLE_DEVICES >> log.txt;
echo $GPUS >> log.txt;
echo $DGCNN_DIR/bin/dgcnn.py inference -bs $GPUS -mbs 1 -np -1 --gpus $CUDA_VISIBLE_DEVICES -rs 1 -it $ITERATION -ld log -kv $KVALUE -ecl $NUM_LAYERS -ecf $NUM_FILTERS -io larcv -dkey data -lkey segment -if $DATA_DIR/$TEST_FILE -mp $WEIGHT_FILE -mn $MODEL_NAME -sh 0 >> log.txt;
$DGCNN_DIR/bin/dgcnn.py inference -bs $GPUS -mbs 1 -np -1 --gpus $CUDA_VISIBLE_DEVICES -rs 1 -it $ITERATION -ld log -kv $KVALUE -ecl $NUM_LAYERS -ecf $NUM_FILTERS -io larcv -dkey data -lkey segment -if $DATA_DIR/$TEST_FILE -mp $WEIGHT_FILE -mn $MODEL_NAME -sh 0 >> log.txt;
cd ..;
mkdir -p $OUT_STORAGE_DIR;
cp -r $WORK_DIR $OUT_STORAGE_DIR;
if [ $? -eq 1 ]; then
cp -r $WORK_DIR $HOME;
fi
if [ $? -eq 0 ]; then
rm -rf $WORK_DIR;
fi
rm -rf $DATA_DIR;
