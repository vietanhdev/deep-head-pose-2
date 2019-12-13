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
    '-t',
    '--train_data_folder', default="./data/300W_LP_prepared/",
    help='Train data directory')
parser.add_argument(
    '-v',
    '--val_data_folder', default="./data/BIWI_prepared/",
    help='Validation data directory')
parser.add_argument(
    '-e',
    '--eval_data_folder', default="./data/BIWI_prepared/",
    help='Test data directory')
parser.add_argument(
    '-m',
    '--model_file', default="./models/shuffle_net_dhp.h5",
    help='Output model file')
args = parser.parse_args()

BIN_NUM = 66
INPUT_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 140

# Prepare dataset
train_dataset = datasets.DataSequence(args.train_data_folder, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, shuffle=True, augment=True)
val_dataset = datasets.DataSequence(args.val_data_folder, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, shuffle=True, augment=True)
test_dataset = datasets.DataSequence(args.eval_data_folder, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, shuffle=True, augment=True)

# Build model
net = models.HeadPoseNet(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, bin_num=BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE, learning_rate=0.001)

# Train model
net.train(args.model_file, max_epoches=EPOCHS, load_weight=False, tf_board_log_dir="./logs")

# Test model
net.test()
