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
parser.add_argument(
    '-s',
    '--show_result', default=False,
    help='Show test resut')
args = parser.parse_args()

# Open and load the config json
with open(args.conf_file) as config_buffer:
    config = json.loads(config_buffer.read())

# Prepare dataset
test_dataset = datasets.DataSequence(config["test"]["test_data_folder"], batch_size=config["test"]["test_batch_size"], input_size=(
    config["model"]["im_width"], config["model"]["im_height"]), shuffle=False, augment=False, random_flip=False)

# Build model
net = models.HeadPoseNet(config["model"]["im_width"], config["model"]
                         ["im_height"], nb_bins=config["model"]["nb_bins"],
                         learning_rate=config["train"]["learning_rate"],
                         backbond=config["model"]["backbond"])

# Load model
net.load_weights(config["test"]["model_file"])

# Test model
net.test(test_dataset, show_result=args.show_result)
