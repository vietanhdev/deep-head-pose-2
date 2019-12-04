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


parser = argparse.ArgumentParser()
parser.add_argument(
    '-s',
    '--data_dir', default="./data/kinect_head_pose_db/hpdb/",
    help='Data directory')

args = parser.parse_args()

filenames = utils.get_list_from_filenames(
    os.path.join(args.data_dir, "filename_list.txt"))
filenames = sorted(filenames)

gpuid = 0
face_detector = RetinaFace('./RetinaFace/retinaface-R50', 0, gpuid, 'net3')

idx = -1
filenames_filtered = []
while True:

    idx += 1
    print(idx)
    if idx < 0:
        idx = 0
    if idx >= len(filenames):
        break

    image_path = os.path.join(
        args.data_dir, "{}_rgb.png".format(filenames[idx]))
    bbox_path = os.path.join(
        args.data_dir, "{}_bbox.txt".format(filenames[idx]))
    landmark_path = os.path.join(
        args.data_dir, "{}_landmark.txt".format(filenames[idx]))
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

    count = 1
    faces, landmarks = face_detector.detect(
        img, 0.8, scales=scales, do_flip=flip)

    draw = img.copy()

    if faces is None or faces.shape[0] == 0:
        continue

    bbox = faces[0]
    bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

    if bbox is not None and all(x > 0 for x in bbox):
        draw = cv2.rectangle(
            draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)

    landmark = landmarks[0].astype(np.int)
    for l in range(landmark.shape[0]):
        color = (0, 0, 255)
        if l == 0 or l == 3:
            color = (0, 255, 0)
        cv2.circle(
            draw, (landmark[l][0], landmark[l][1]), 1, color, 2)

    cv2.imshow("Image", draw)
    k = cv2.waitKey(1)

    filenames_filtered.append(filenames[idx])

    with open(bbox_path, "w") as text_file:
        text_file.write(
            " ".join(map(str, [bbox[0], bbox[1], bbox[2], bbox[3]])))

    with open(landmark_path, "w") as text_file:
        points = []
        for l in range(landmark.shape[0]):
            points.append("{} {}".format(landmark[l][0], landmark[l][1]))
        text_file.write("\n".join(points))



# Save filtered file list
with open(os.path.join(args.data_dir, "filename_list_filtered.txt"), "w") as text_file:
    text_file.write(
        "\n".join(filenames_filtered))
