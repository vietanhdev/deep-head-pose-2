import os
import sys
import random
import cv2
import numpy as np
from tqdm import tqdm
import json
import pathlib

def write_data_folder(examples, output_folder):

    # Make output folder
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    for i, example in enumerate(tqdm(examples)):
        try:

            # Crop face
            image = cv2.imread(example['image_path'])
            bbox = example["face_bbox"]
            crop = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            cv2.imwrite(os.path.join(output_folder, str(i) + '.png'))

            # Bin values
            bins = list(range(-99, 99, 3))
            binned_labels = np.digitize(
                [example['yaw'],  example['pitch'],  example['roll']], bins) - 1

            # Pose label
            yaw = [example['yaw'], binned_labels[0]]
            pitch = [example['pitch'], binned_labels[1]]
            roll = [example['roll'], binned_labels[2]]

            # Write label
            label = {
                'image': str(i) + '.png',
                'roll': roll,
                'pitch': pitch,
                'yaw': yaw,
                'landmark': example['landmark']
            }

            with open(os.path.join(output_folder, str(i) + '.json'), 'w') as outfile:
                json.dump(label, outfile)
        except Exception as inst:
            print(inst)
            pass