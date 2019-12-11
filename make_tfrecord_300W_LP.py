import argparse
from pathlib import Path
import os
import numpy as np
import cv2
import time
import utils
from imutils import face_utils
import scipy.io as sio

import sys
import datetime
import glob

from dataset_utils import *

import pathlib

parser = argparse.ArgumentParser()

parser.add_argument(
    '-d',
    '--data_dir', default="./data/300W_LP",
    help='Data directory')

parser.add_argument(
    '-s',
    '--input_size', default=128,
    type=int,
    help='Input size for deep head pose')
    
parser.add_argument(
    '-o',
    '--output_file', default="./data/300W_LP.tfrecord",
    help='Output file')

args = parser.parse_args()

def get_pt2d_from_mat(mat_path):
    # Get 2D landmarks
    mat = sio.loadmat(mat_path)
    pt2d = mat['pt2d']
    return pt2d
def get_ypr_from_mat(mat_path):
    # Get yaw, pitch, roll from .mat annotation.
    # They are in radians
    mat = sio.loadmat(mat_path)
    # [pitch yaw roll tdx tdy tdz scale_factor]
    pre_pose_params = mat['Pose_Para'][0]
    # Get [pitch, yaw, roll]
    pose_params = pre_pose_params[:3]
    return pose_params


# Create file list
filenames = []
for root, dirs, files in os.walk(args.data_dir):
    for filename in files:
        path = os.path.relpath(os.path.join(root, filename), args.data_dir)
        if path.endswith(".jpg"):
            filenames.append(path[:-4])
filenames = sorted(filenames)


idx = -1
n_prrocessed = 0
examples = []
while True:

    idx += 1
    print("Processed: {}/{}".format(n_prrocessed, idx))
    
    if idx < 0:
        idx = 0
    if idx >= len(filenames):
        break

    # Skip Flip examples
    if "_Flip" in filenames[idx]:
        continue

    example = {}

    image_path = os.path.join(
        args.data_dir, "{}.jpg".format(filenames[idx]))
    example["image_path"] = image_path

    # Read original image
    img = cv2.imread(image_path)
    draw = img.copy()

    # Label
    label = {}

    # Crop the face loosely
    mat_path = os.path.join(args.data_dir, "landmarks", filenames[idx] + "_pts.mat")
    ldm_mat = sio.loadmat(mat_path)
    landmark_64p = ldm_mat["pts_2d"]

    landmark = []
    point0_x = int((landmark_64p[37, 0] + landmark_64p[40, 0]) / 2)
    point0_y = int((landmark_64p[37, 1] + landmark_64p[40, 1]) / 2)
    landmark.append({'x': point0_x, 'y': point0_y})
    point1_x = int((landmark_64p[43, 0] + landmark_64p[46, 0]) / 2)
    point1_y = int((landmark_64p[43, 1] + landmark_64p[46, 1]) / 2)
    landmark.append({'x': point1_x, 'y': point1_y})
    landmark.append({'x': landmark_64p[30, 0], 'y': landmark_64p[30, 1]})
    landmark.append({'x': landmark_64p[48, 0], 'y': landmark_64p[48, 1]})
    landmark.append({'x': landmark_64p[64, 0], 'y': landmark_64p[64, 1]})
    
    for i in range(len(landmark)):
        x = int(landmark[i]['x'])
        y = int(landmark[i]['y'])

        draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(draw, (int(x), int(y)), 1, (0,255,0))
    

    pt2d = get_pt2d_from_mat(os.path.join(args.data_dir, filenames[idx] + ".mat"))
    x_min = int(min(pt2d[0, :]))
    y_min = int(min(pt2d[1, :]))
    x_max = int(max(pt2d[0, :]))
    y_max = int(max(pt2d[1, :]))
    y_min = max(0, y_min - int(0.3 * (y_max - y_min))) # Extend face to top


    # Crop face
    bbox = (x_min, y_min, x_max, y_max)
    bbox_loosen, scale_x, scale_y = utils.get_loose_bbox(bbox, img, args.input_size)
    crop = utils.crop_face_loosely(bbox, img, args.input_size)
    example["face_bbox"] = bbox_loosen

    # Adjust landmark points
    example["landmark"] = []
    points = []
    for l in range(len(landmark)):

        # Normalize landmark points
        x = float(landmark[l]['x'] - bbox_loosen[0]) * scale_x
        y = float(landmark[l]['y'] - bbox_loosen[1]) * scale_y
        x -= args.input_size // 2
        y -= args.input_size // 2
        x /= args.input_size
        y /= args.input_size

        example["landmark"].append({'x': x, 'y': y})

    # We get the pose in radians
    pose = get_ypr_from_mat(os.path.join(args.data_dir, filenames[idx] + ".mat"))
    # And convert to degrees.
    pitch = pose[0] * 180 / np.pi
    yaw = pose[1] * 180 / np.pi
    roll = pose[2] * 180 / np.pi

    example["roll"] = roll
    example["yaw"] = yaw
    example["pitch"] = pitch
    
    examples.append(example)
    n_prrocessed += 1

random.seed(42)
random.shuffle(examples)

write_tfrecord(examples, args.output_file)
