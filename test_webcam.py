import argparse
from RetinaFace.retinaface import RetinaFace
import models
import utils
import datasets
from imutils import face_utils
import scipy.io as sio
import cv2
import numpy as np
import os
import json
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.compat.v2.config.experimental.set_memory_growth(gpu, True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    '--model_file', default="./models/shuffle_net_dhp.h5",
    help='Output model file')
parser.add_argument(
    '-c',
    '--conf_file', default="config.json",
    help='Configuration file')
args = parser.parse_args()

# Open and load the config json
with open(args.conf_file) as config_buffer:
    config = json.loads(config_buffer.read())

# Build model
net = models.HeadPoseNet(config["model"]["im_width"], config["model"]
                         ["im_height"], nb_bins=config["model"]["nb_bins"],
                         learning_rate=config["train"]["learning_rate"],
                         backbond=config["model"]["backbond"])
# Load model
net.load_weights(config["test"]["model_file"])

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

            face_crops = []
            face_boxes = []
            for i in range(len(faces)):
                bbox = faces[i]
                bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                face_crop = utils.crop_face_loosely(bbox, frame, (config["model"]["im_width"], config["model"]["im_height"]))
                face_crop = cv2.resize(face_crop, (config["model"]["im_width"], config["model"]["im_height"]))
                face_box, _, _ = utils.get_loosen_bbox(bbox, frame, (config["model"]["im_width"], config["model"]["im_height"]))
                face_boxes.append(face_box)

                # Convert crop image to RGB color space
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            
                face_crops.append(face_crop)
                cv2.imshow("crop", face_crop)

            if len(face_crops) > 0:

                batch_yaw, batch_pitch, batch_roll, batch_landmark = net.predict_batch(
                    face_crops)

                draw = frame.copy()

                for i in range(batch_yaw.shape[0]):
                    yaw = batch_yaw[i]
                    pitch = batch_pitch[i]
                    roll = batch_roll[i]
                    landmark = batch_landmark[i]
                    face_box_size = (face_boxes[i][2]-face_boxes[i][0], face_boxes[i][3]-face_boxes[i][1])
                    net_input_size = (config["model"]["im_width"], config["model"]["im_height"])
                    scale = np.divide(np.array(face_box_size), np.array(net_input_size))

                    draw = cv2.rectangle(draw, (face_boxes[i][0], face_boxes[i][1]), (
                        face_boxes[i][2], face_boxes[i][3]), (0, 0, 255), 2)

                    face_box_width = face_boxes[i][2]-face_boxes[i][0]
                    axis_x, axis_y = utils.unnormalize_landmark_point(
                        (landmark[4], landmark[5]), net_input_size, scale=scale)
                    axis_x += face_boxes[i][0]
                    axis_y += face_boxes[i][1]
                    draw = utils.draw_axis(draw, yaw, pitch, roll, tdx=face_boxes[i][0] + face_box_width // 5,
                                           tdy=face_boxes[i][1] + face_box_width // 5, size=face_box_width // 2)

                    for j in range(5):
                        x = landmark[2 * j]
                        y = landmark[2 * j + 1]
                        x, y = utils.unnormalize_landmark_point(
                            (x, y), net_input_size, scale=scale)
                        x += face_boxes[i][0]
                        y += face_boxes[i][1]
                        x = int(x)
                        y = int(y)
                        draw = cv2.circle(draw, (x, y), 2, (255, 0, 0), 2)

                cv2.imshow("Result", draw)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
