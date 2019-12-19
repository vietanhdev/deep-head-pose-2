import json
import argparse
from pathlib import Path
import models
import utils
import datasets
from imutils import face_utils
import scipy.io as sio
import cv2
import numpy as np
import os
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-c',
    '--conf_file', default="config.json",
    help='Configuration file')
args = parser.parse_args()

# Open and load the config json
with open(args.conf_file) as config_buffer:
    config = json.loads(config_buffer.read())

# Prepare dataset
train_dataset = datasets.DataSequence(config["train"]["train_data_folder"], batch_size=config["train"]["train_batch_size"], input_size=(
    config["model"]["im_width"], config["model"]["im_height"]), shuffle=True, augment=True, random_flip=True)
val_dataset = datasets.DataSequence(config["train"]["val_data_folder"], batch_size=config["train"]["val_batch_size"], input_size=(
    config["model"]["im_width"], config["model"]["im_height"]), shuffle=True, augment=True, random_flip=True)
test_dataset = datasets.DataSequence(config["test"]["test_data_folder"], batch_size=config["test"]["test_batch_size"], input_size=(
    config["model"]["im_width"], config["model"]["im_height"]), shuffle=False, augment=False, random_flip=False)

# Build model
net = models.HeadPoseNet(config["model"]["im_width"], config["model"]
                        ["im_height"], nb_bins=config["model"]["nb_bins"], learning_rate=config["train"]["learning_rate"], loss_weights=config["train"]["loss_weights_yaw_pitch_roll_landmark"],
                        loss_angle_alpha=config["train"]["loss_angle_alpha"],
                        backbond=config["model"]["backbond"])

# Train model
net.train(train_dataset, val_dataset, config["train"])

# Test model
net.test(test_dataset)
