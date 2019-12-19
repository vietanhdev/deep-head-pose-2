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
    '--data_dir', default="./data/AFLW2000",
    help='Data directory')

parser.add_argument(
    '-s',
    '--input_size', default=128,
    type=int,
    help='Input size for deep head pose')
    
parser.add_argument(
    '-o',
    '--output_folder', default="./data/AFLW2000_prepared/",
    help='Output folder')

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
    # ldm_mat = sio.loadmat(os.path.join(args.data_dir, filenames[idx] + ".mat"))
    # print(ldm_mat)
    pt2d = get_pt2d_from_mat(os.path.join(args.data_dir, filenames[idx] + ".mat"))
    print(pt2d.shape)

    landmark = []
    landmark.append({'x': pt2d[:, 7][0], 'y': pt2d[:, 7][1]})
    landmark.append({'x': pt2d[:, 10][0], 'y': pt2d[:, 10][1]})
    landmark.append({'x': pt2d[:, 14][0], 'y': pt2d[:, 14][1]})
    landmark.append({'x': pt2d[:, 17][0], 'y': pt2d[:, 17][1]})
    landmark.append({'x': pt2d[:, 19][0], 'y': pt2d[:, 19][1]})


    def show_image_with_landmark(window_name, image, landmark, confirm=False):
        for i in range(len(landmark)):
            x = int(landmark[i]['x'])
            y = int(landmark[i]['y'])
            image = cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,  
                    1, (255, 0, 0), 1, cv2.LINE_AA)
            image = cv2.circle(image, (int(x), int(y)), 2, (0,255,0))

        if confirm:
            # Using cv2.putText() method 
            image = cv2.putText(image, 'Confirm?', (100, 100) , cv2.FONT_HERSHEY_SIMPLEX ,  
                   1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, image)

        return image

    def re_label(img):
        print("Re-labeling image")
        while True:
            draw = img.copy()
            landmark = []
            for i in range(5): 
                r = cv2.selectROI("Label", draw)
                landmark.append({'x': r[0], 'y': r[1]})
                draw = show_image_with_landmark("Label", draw, landmark)
            draw = show_image_with_landmark("Label", draw, landmark, confirm=True)
            r = cv2.selectROI("Label", draw)
            if r == (0,0,0,0):
                return landmark
    
    for i in range(len(landmark)):
        x = int(landmark[i]['x'])
        y = int(landmark[i]['y'])

        draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,  
                   0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(draw, (int(x), int(y)), 1, (0,0,255))

    # Fill missing points
    if landmark[0]['x'] == -1 or landmark[1]['x'] == -1 or landmark[3]['x'] == -1 or landmark[4]['x'] == -1:
        landmark = re_label(draw.copy())

    # cv2.imshow("Label", draw)
    # cv2.waitKey(0)

    x_min = int(min(pt2d[0, :]))
    y_min = int(min(pt2d[1, :]))
    x_max = int(max(pt2d[0, :]))
    y_max = int(max(pt2d[1, :]))
    y_min = max(0, y_min - int(0.3 * (y_max - y_min))) # Extend face to top


    # Crop face
    bbox = (x_min, y_min, x_max, y_max)
    bbox_loosen, scale_x, scale_y = utils.get_loosen_bbox(bbox, img, (args.input_size, args.input_size))
    crop = utils.crop_face_loosely(bbox, img, (args.input_size, args.input_size))
    example["face_bbox"] = bbox_loosen

    # Adjust landmark points
    example["landmark"] = []
    points = []
    for l in range(len(landmark)):

        # Normalize landmark points
        x = float(landmark[l]['x'] - bbox_loosen[0]) * scale_x
        y = float(landmark[l]['y'] - bbox_loosen[1]) * scale_y
        x, y = utils.normalize_landmark_point((x, y), (args.input_size, args.input_size))

        example["landmark"].append([x, y])

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

write_data_folder(examples, args.output_folder)
