import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
import os
import numpy as np
import cv2
import scipy.io as sio
from imutils import face_utils

import datasets
import utils
import models

from pathlib import Path
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
args = parser.parse_args()

BIN_NUM = 66
INPUT_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 140

# Prepare dataset
dataset = datasets.Biwi(args.data_dir, 'filename_list_filtered.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, train_ratio=0.8, val_ratio=0.15)

# Build model
net = models.HeadPoseNet(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, learning_rate=0.001)

# Train model
net.train(args.model_file, max_epoches=EPOCHS, load_weight=False, tf_board_log_dir="./logs")

# Test model
net.test()
