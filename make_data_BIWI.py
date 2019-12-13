import argparse
from pathlib import Path
import os
import numpy as np
import cv2
import time
import utils
from imutils import face_utils

import sys
import datetime
import glob
from RetinaFace.retinaface import RetinaFace

import random

from dataset_utils import *

import pathlib

parser = argparse.ArgumentParser()

parser.add_argument(
    '-d',
    '--data_dir', default="./data/BIWI/kinect_head_pose_db/hpdb/",
    help='Data directory')

parser.add_argument(
    '-s',
    '--input_size', default=128,
    type=int,
    help='Input size for deep head pose')
    
parser.add_argument(
    '-o',
    '--output_folder', default="./data/BIWI_prepared/",
    help='Output file')

args = parser.parse_args()


# Input image file list
filenames = utils.get_list_from_filenames(
    os.path.join(args.data_dir, "filename_list.txt"))
filenames = sorted(filenames)
random.shuffle(filenames)

# Init face detector
gpuid = 0
face_detector = RetinaFace('./RetinaFace/retinaface-R50', 0, gpuid, 'net3')

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

    example = {}

    image_path = os.path.join(
        args.data_dir, "{}_rgb.png".format(filenames[idx]))
    example["image_path"] = image_path

    # Read original image
    img = cv2.imread(image_path)

    # Cover unrelated faces
    if "02/frame" in image_path or "04/frame" in image_path:
        img[100:180, :260] = [0, 0, 0]
    if "07/frame" in image_path:
        img[100:180, :100] = [0, 0, 0]
    if "09/frame" in image_path or "10/frame" in image_path:
        img[100:260, :150] = [0, 0, 0]

    # Retina Face
    scales = [1.0]
    flip = False
    faces, landmarks = face_detector.detect(
        img, 0.8, scales=scales, do_flip=flip)
    
    if faces is None or faces.shape[0] == 0:
        continue

    # Label
    label = {}

    # Crop face
    bbox = faces[0]
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    bbox_loosen, scale_x, scale_y = utils.get_loosen_bbox(bbox, img, args.input_size)
    crop = img[bbox_loosen[1]:bbox_loosen[3], bbox_loosen[0]:bbox_loosen[2]]
    crop = cv2.resize(crop, (args.input_size, args.input_size))

    draw = crop.copy()
    example["face_bbox"] = bbox_loosen

    # Landmark
    example["landmark"] = []
    landmark = landmarks[0].astype(np.int)
    points = []
    for l in range(landmark.shape[0]):

        # Normalize landmark points
        x = float(landmark[l][0] - bbox_loosen[0]) * scale_x
        y = float(landmark[l][1] - bbox_loosen[1]) * scale_y
        
        draw = cv2.putText(draw, str(l), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.circle(draw, (int(x), int(y)), 1, (0,255,0))

        x, y = utils.normalize_landmark_point((x, y), (args.input_size, args.input_size))

        example["landmark"].append([x, y])

    # cv2.imshow("Crop", draw)
    # cv2.waitKey(0)

    # Load pose
    pose_path = os.path.join(args.data_dir, filenames[idx] + '_pose.txt')
    with open(pose_path, 'r') as pose_annot:
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)
        
        R = np.array(R)
        T = R[3, :]
        R = R[:3, :]
        pose_annot.close()
        
        R = np.transpose(R)
        
        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi

        example["roll"] = roll
        example["yaw"] = yaw
        example["pitch"] = pitch
    
        
    examples.append(example)
    n_prrocessed += 1

random.seed(42)
random.shuffle(examples)

write_data_folder(examples, args.output_folder, image_size=(args.input_size, args.input_size))
