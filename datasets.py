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

    def __init__(self, data_folder, batch_size=8, input_size=(128, 128), shuffle=True, augment=False, random_flip=True, normalize=True):
        """
        Keras Sequence object to train a model on larger-than-memory data.
        """

        self.batch_size = batch_size
        self.input_size = input_size
        self.image_files = self.__get_image_files(data_folder)
        self.file_num = len(self.image_files)
        self.data_folder = data_folder
        self.random_flip = random_flip
        self.augment = augment
        self.normalize = normalize

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
            
            # Load image
            # Flip 50% of images
            flip = False
            if self.random_flip and random.random() < 0.5:
                flip = True
            image, label = self.__get_input_img(image_file, label=label,  augment=self.augment, flip=flip)

            # Add binned value for head pose
            bins = list(range(-99, 99, 3))
            binned_labels = np.digitize(
                [label['yaw'],  label['pitch'],  label['roll']], bins) - 1
            yaw = [label['yaw'], binned_labels[0]]
            pitch = [label['pitch'], binned_labels[1]]
            roll = [label['roll'], binned_labels[2]]

            batch_image.append(image)
            batch_yaw.append(yaw)
            batch_pitch.append(pitch)
            batch_roll.append(roll)
            batch_landmark.append(label['landmark'])

        batch_image = np.array(batch_image)
        batch_landmark = np.array(batch_landmark)
        batch_landmark = batch_landmark.reshape(batch_landmark.shape[0], -1)
        batch_yaw = np.array(batch_yaw)
        batch_pitch = np.array(batch_pitch)
        batch_roll = np.array(batch_roll)

        return batch_image, [batch_yaw, batch_pitch, batch_roll, batch_landmark]

    def set_normalization(self, normalize):
        self.normalize = normalize

    def __get_image_files(self, data_folder):
        image_files = os.listdir(data_folder)
        image_files = [os.path.join(data_folder, f) for f in image_files if f.lower().endswith(".jpg") or f.lower().endswith(".png")]
        return image_files

    def __get_input_img(self, file_name, label, augment=False, flip=False):

        if flip:
            label['yaw'] = -label['yaw']
            label['roll'] = -label['roll']
            label["landmark"] = np.multiply(label["landmark"], np.array([-1, 1]))

        unnomarlized_landmark = utils.unnormalize_landmark(label["landmark"], self.input_size)
        img = cv2.imread(file_name)

        if flip:
            img = cv2.flip(img, 1)
        
        if augment:
            img, unnomarlized_landmark = augment_img(img, unnomarlized_landmark)

        label["landmark"] = utils.normalize_landmark(unnomarlized_landmark, self.input_size)

        # Uncomment following lines to write out augmented images for debuging
        # cv2.imwrite("aug_" + str(random.randint(0, 50)) + ".png", img)
        # cv2.waitKey(0)

        # draw = img.copy()
        # unnomarlized_landmark = utils.unnormalize_landmark(label["landmark"], self.input_size)
        # for i in range(len(unnomarlized_landmark)):
        #     x = int(unnomarlized_landmark[i][0])
        #     y = int(unnomarlized_landmark[i][1])

        #     draw = cv2.putText(draw, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,  
        #             0.5, (255, 255, 255), 1, cv2.LINE_AA)
        #     cv2.circle(draw, (int(x), int(y)), 1, (0,0,255))

        # cv2.imshow("draw", draw)
        # cv2.waitKey(0)

        if self.normalize:
            img = (img - img.mean())/img.std()

        return img, label

    def __get_input_label(self, file_name):
        with open(file_name) as json_file:
            data = json.load(json_file)
        return data