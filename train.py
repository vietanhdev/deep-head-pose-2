import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import dlib
from imutils import face_utils

import datasets
import utils
import models

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-s',
    '--data_dir', default="./data/kinect_head_pose_db/hpdb/",
    help='Data directory')
parser.add_argument(
    '-m',
    '--model_file', default="./models/shuffle_net_dhp.h5",
    help='Output model file')
parser.add_argument(
    '-t',
    '--test_save_dir', default="./data/kinect_head_pose_test/",
    help='Test save directory')
args = parser.parse_args()

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE=16
EPOCHS=20

# Prepare dataset
dataset = datasets.Biwi(args.data_dir, 'filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, ratio=0.95)

# Build model
net = models.HeadPoseNet(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

# Train model
net.train(args.model_file, max_epoches=EPOCHS, load_weight=False)

# Test model
net.test(args.test_save_dir)
