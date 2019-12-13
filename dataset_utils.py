import os
import sys
import random
import cv2
import numpy as np
from tqdm import tqdm
import json
import pathlib

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def write_data_folder(examples, output_folder, image_size=(128,128)):

    # Make output folder
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    for i, example in enumerate(tqdm(examples)):
        try:

            # Crop face
            image = cv2.imread(example['image_path'])
            bbox = example["face_bbox"]
            crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            crop = cv2.resize(crop, image_size)

            cv2.imwrite(os.path.join(output_folder, str(i) + '.png'), crop)

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
                json.dump(label, outfile, cls=NpEncoder)
        except Exception as inst:
            print(inst)
            pass