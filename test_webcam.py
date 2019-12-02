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


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    '--model_file', default="./models/shuffle_net_dhp.h5",
    help='Output model file')
args = parser.parse_args()

face_landmark_path = 'models/shape_predictor_68_face_landmarks.dat'

BIN_NUM = 66
INPUT_SIZE = 64
BATCH_SIZE=16

net = models.HeadPoseNet(None, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)
net.train(args.model_file, load_weight=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")
    exit(-1)
    
    
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        face_rects = detector(frame, 0)
        if len(face_rects) > 0:
            shape = predictor(frame, face_rects[0])
            shape = face_utils.shape_to_np(shape)

            face_crop = utils.crop_face_loosely(shape, frame, INPUT_SIZE)
            
            frames.append(face_crop)
            if len(frames) == 1:
                print(shape[30])
                pred_cont_yaw, pred_cont_pitch, pred_cont_roll = net.test_online(frames)
                
                cv2_img = utils.draw_axis(frame, pred_cont_yaw, pred_cont_pitch, pred_cont_roll, tdx=shape[30][0],
                                          tdy=shape[30][1], size=100)
                cv2.imshow("cv2_img", cv2_img)
                frames = []
                
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break