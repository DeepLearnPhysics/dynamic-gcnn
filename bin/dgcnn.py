#!/usr/bin/python
import os,sys
DGCNN_DIR = os.path.dirname(os.path.abspath(__file__))
DGCNN_DIR = os.path.dirname(DGCNN_DIR)
sys.path.insert(0,DGCNN_DIR)
from dgcnn import DGCNN_FLAGS

def main():
  flags = DGCNN_FLAGS()
  flags.parse_args()  

if __name__ == '__main__':
  main()

