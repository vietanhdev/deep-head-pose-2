import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import random
import textwrap
from tensorflow.keras.utils import Sequence
import math
from augmentation import augment_img
import random
import glob
import json
from augmentation import augment_img

class DataSequence(Sequence):

    def __init__(self, data_folder, batch_size=8, input_size=128, shuffle=True, augment=False):
        """
        Keras Sequence object to train a model on larger-than-memory data.
        """

        self.batch_size = batch_size
        self.input_size = input_size
        self.image_files = self.__get_image_files(data_folder)
        self.file_num = len(self.image_files)
        self.data_folder = data_folder
        self.augment = augment

        if shuffle:
            random.shuffle(self.image_files)
       
    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return int(math.ceil(len(self.image_files) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """

        batch_image_files = self.image_files[idx * self.batch_size: (1 + idx) * self.batch_size]

        batch_image = []
        batch_yaw = []
        batch_pitch = []
        batch_roll = []
        batch_landmark = []

        for image_file in batch_image_files:
            
            # Read images and labels
            label = self.__get_input_label(image_file.replace(".png", ".json"))
            landmark = label['landmark']
            unnomarlized_landmark = utils.unnormalize_landmark(landmark, (self.input_size,self.input_size))
            image, unnomarlized_landmark = self.__get_input_img(image_file, augment=self.augment, landmark=unnomarlized_landmark)
            landmark = utils.normalize_landmark(landmark, (self.input_size,self.input_size))

            # Landmark
            batch_image.append(image)
            batch_yaw.append([label['yaw'][1], label['yaw'][0]])
            batch_pitch.append([label['pitch'][1], label['pitch'][0]])
            batch_roll.append([label['roll'][1], label['roll'][0]])
            batch_landmark.append(landmark)

        batch_image = np.array(batch_image, dtype=np.float32)
        batch_landmark = np.array(batch_landmark)
        batch_landmark = batch_landmark.reshape(batch_landmark.shape[0], -1)
        batch_yaw = np.array(batch_yaw)
        batch_pitch = np.array(batch_pitch)
        batch_roll = np.array(batch_roll)

        return batch_image, [batch_yaw, batch_pitch, batch_roll, batch_landmark]

    def __get_image_files(self, data_folder):
        image_files = os.listdir(data_folder)
        image_files = [os.path.join(data_folder, f) for f in image_files if f.lower().endswith(".jpg") or f.lower().endswith(".png")]
        return image_files

    def __get_input_img(self, file_name, augment=False, landmark=[]):
        img = cv2.imread(file_name)
        if augment:
            img, landmark = augment_img(img, landmark)
            # Uncomment following lines to write out augmented images for debuging
            # cv2.imwrite("aug_" + str(random.randint(0, 50)) + ".png", img)
            # cv2.waitKey(0)
        normed_img = (img - img.mean())/img.std()
        return normed_img, landmark

    def __get_input_label(self, file_name):
        with open(file_name) as json_file:
            data = json.load(json_file)
        return data