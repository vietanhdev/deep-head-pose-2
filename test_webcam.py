import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import dlib
from imutils import face_utils

import datasets
import utils
import models
import utils

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    '--model_file', default="./models/shuffle_net_dhp.h5",
    help='Output model file')
args = parser.parse_args()

BIN_NUM = 66
INPUT_SIZE = 128
BATCH_SIZE=16

net = models.HeadPoseNet(None, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)
net.train(args.model_file, load_weight=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")
    exit(-1)
    
    
face_detector = dlib.get_frontal_face_detector()


while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_rects = face_detector(frame, 0)
        if len(face_rects) > 0:

            face = face_rects[0]
            bbox = (face.left(), face.top(), face.right(), face.bottom())

            face_crop = utils.crop_face_loosely(bbox, frame, INPUT_SIZE)
            face_box, _, _ = utils.get_loose_bbox(bbox, frame, INPUT_SIZE)
            
            frames = []
            frames.append(face_crop)
            if len(frames) == 1:
                # print(shape[30])
                pred_cont_yaw, pred_cont_pitch, pred_cont_roll, landmark = net.test_online(frames)
                print((pred_cont_yaw, pred_cont_pitch, pred_cont_roll))

                cv2_img = utils.draw_axis(frame, pred_cont_yaw, pred_cont_pitch, pred_cont_roll, tdx=bbox[0],
                                          tdy=bbox[1], size=100)

                x = landmark[0]
                y = landmark[1]
                print(face_box)
                x, y = utils.get_original_landmark_point(x, y, face_box, 128)
                print((x, y))
                cv2_img = cv2.circle(cv2_img, (x+128, y+128), 2, (255, 0, 0), 2)

                cv2.imshow("cv2_img", cv2_img)
                frames = []
                
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break