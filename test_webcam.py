import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
from imutils import face_utils

import datasets
import utils
import models
import utils
from RetinaFace.retinaface import RetinaFace

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    '--model_file', default="./models/shuffle_net_dhp_test18.h5",
    help='Output model file')
args = parser.parse_args()

BIN_NUM = 66
INPUT_SIZE = 128
BATCH_SIZE = 16

net = models.HeadPoseNet(None, BIN_NUM, batch_size=BATCH_SIZE, input_size=INPUT_SIZE)
net.train(args.model_file, load_weight=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")
    exit(-1)
    
    
face_detector = RetinaFace('./RetinaFace/retinaface-R50', 0, 0, 'net3')

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        faces, landmarks = face_detector.detect(
            frame, 0.8, scales=[1.0], do_flip=False)

        if len(faces) > 0:
            bbox = faces[0]
            bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

            face_crop = utils.crop_face_loosely(bbox, frame, INPUT_SIZE)
            face_crop = np.asarray(face_crop)
            normed_face_crop = (face_crop - face_crop.mean())/face_crop.std()
            face_box, _, _ = utils.get_loose_bbox(bbox, frame, INPUT_SIZE)
            
            frames = []
            frames.append(normed_face_crop)
            if len(frames) == 1:
                # print(shape[30])
                pred_cont_yaw, pred_cont_pitch, pred_cont_roll, landmark = net.test_online(frames)

                draw = cv2.rectangle(frame, (face_box[0], face_box[1]), (face_box[2], face_box[3]), (0,0,255), 2)
                draw = utils.draw_axis(draw, pred_cont_yaw, pred_cont_pitch, pred_cont_roll, tdx=bbox[0],
                                          tdy=bbox[1], size=100)


                for i in range(5):
                    x = landmark[2 * i]
                    y = landmark[2 * i + 1]
                    x, y = utils.get_original_landmark_point(x, y, face_box, 128)
                    draw = cv2.circle(draw, (x, y), 2, (255, 0, 0), 2)

                cv2.imshow("Result", draw)
                frames = []
                
            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break