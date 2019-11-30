import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)
import os
import numpy as np
import cv2
import scipy.io as sio
import dlib
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
    '--model_file', default="./models/shuffle_net_dhp",
    help='Output model file')
parser.add_argument(
    '-t',
    '--test_save_dir', default="./test_result/",
    help='Test save directory')
args = parser.parse_args()

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 30

# Prepare dataset
dataset = datasets.Biwi(args.data_dir, 'filename_list.txt', batch_size=BATCH_SIZE, input_size=INPUT_SIZE, train_ratio=0.8, val_ratio=0.15)

# Build model
net = models.HeadPoseNet(dataset, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)

# Train model
net.train(args.model_file, max_epoches=EPOCHS, load_weight=False, tf_board_log_dir="./logs")

# Test model
Path(args.test_save_dir).mkdir(parents=True, exist_ok=True) # Make test folder if not exist
net.test(args.test_save_dir)
