# dgcnn

This is an implementation of 3D point cloud semantic segmentation for [Dynamic Graph Convolutional Neural Network](https://arxiv.org/abs/1801.07829). **[WARNING:] this impelementation is still on-going** so watch out if you want to use it. This warning sentence will be removed when ready with reproducible result with open data sample.

### Requirements
* `tensorflow >= v1.3`
* `numpy >= 1.13` 
* Optional requirements for IO include `h5py`, `larcv`

### Help
An executable script can be found at `bin/dgcnn.py`. The script takes `train` or `inference` arguments. Try `--help` to list available arguments:
```
bin/dgcnn.py train --help
```
### How to run
Below is an example of how to train the network using `mydata.hdf5` data file with `hdf5` format, 4 GPUs with batch size 24 and mini-batch size of 6, store snapshot every 500 iterations, print out info (loss,accuracy,etc) every 10 iterations, and store tensorboard summary every 50 iterations.
```
bin/dgcnn.py train --gpus 0,1,2,3 -bs 24 -mbs 6 -chks 500 -rs 10 -ss 50 -if mydata.hdf5 -io h5
```
See `--help` to find more flags and a descipriton for arguments.



