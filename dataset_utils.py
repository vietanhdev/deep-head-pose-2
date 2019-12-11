import os
import sys
import random

import cv2
import numpy as np
from tqdm import tqdm

import tensorflow as tf


def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def write_tfrecord(examples, output_filename):
    writer = tf.io.TFRecordWriter(output_filename)
    for example in tqdm(examples):
        try:
            
            # Crop face
            image = cv2.imread(example['image_path'])
            bbox = example["face_bbox"]
            crop = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]

            if image is not None:
                encoded_image_string = cv2.imencode('.jpg', crop)[1].tostring()
                feature = {
                    'train/image': bytes_feature(tf.compat.as_bytes(encoded_image_string)),
                    'train/roll': float_feature(example['roll']),
                    'train/pitch': float_feature(example['pitch']),
                    'train/yaw': float_feature(example['yaw']),
                    'train/landmark/0/x': float_feature(example['landmark'][0]['x']),
                    'train/landmark/0/y': float_feature(example['landmark'][0]['y']),
                    'train/landmark/1/x': float_feature(example['landmark'][1]['x']),
                    'train/landmark/1/y': float_feature(example['landmark'][1]['y']),
                    'train/landmark/2/x': float_feature(example['landmark'][2]['x']),
                    'train/landmark/2/y': float_feature(example['landmark'][2]['y']),
                    'train/landmark/3/x': float_feature(example['landmark'][3]['x']),
                    'train/landmark/3/y': float_feature(example['landmark'][3]['y']),
                    'train/landmark/4/x': float_feature(example['landmark'][4]['x']),
                    'train/landmark/4/y': float_feature(example['landmark'][4]['y']),
                }

                tf_example = tf.train.Example(features = tf.train.Features(feature=feature))
                writer.write(tf_example.SerializeToString())
        except Exception as inst:
            print(inst)
            pass
    writer.close()