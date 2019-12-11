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

def split_samples(samples_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.15):
    with open(samples_file) as samples_fp:
        lines = samples_fp.readlines()
        random.shuffle(lines)

        train_num = int(len(lines) * train_ratio)
        val_num = int(len(lines) * val_ratio)
        test_num = len(lines) - train_num - val_num
        count = 0
        data = []
        for line in lines:
            count += 1
            data.append(line)
            if count == train_num:
                with open(train_file, "w+") as train_fp:
                    for d in data:
                        train_fp.write(d)
                data = []

            if count == train_num + val_num:
                with open(val_file, "w+") as val_fp:
                    for d in data:
                        val_fp.write(d)
                data = []

            if count == train_num + val_num + test_num:
                with open(test_file, "w+") as test_fp:
                    for d in data:
                        test_fp.write(d)
                data = []
    return train_num, val_num, test_num
            


class BIWIDataSequence(Sequence):

    def __init__(self, data_dir, sample_file, batch_size, input_size=128, shuffle=True, augment=False):
        """
        Keras Sequence object to train a model on larger-than-memory data.
        """

        self.sample_file = sample_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.filenames = self.__get_list_from_filenames(sample_file)
        self.file_num = len(self.filenames)
        self.data_dir = data_dir
        self.augment = augment

        if shuffle:
            idx = np.random.permutation(range(self.file_num))
            self.filenames = np.array(self.filenames)[idx]
        self.max_num = self.file_num - (self.file_num % self.batch_size)
     
       
    def __len__(self):
        """
        Number of batch in the Sequence.
        :return: The number of batches in the Sequence.
        """
        return int(math.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        Retrieve the mask and the image in batches at position idx
        :param idx: position of the batch in the Sequence.
        :return: batches of image and the corresponding mask
        """

        batch_filenames = self.filenames[idx * self.batch_size: (1 + idx) * self.batch_size]

        batch_x = []
        batch_yaw = []
        batch_pitch = []
        batch_roll = []
        batch_landmark = []

        for filename in batch_filenames:
            img = self.__get_input_img(self.data_dir, filename, augment=self.augment)
            bin_labels, cont_labels = self.__get_input_label(self.data_dir, filename)

            # Load landmark
            landmark_path = os.path.join(
                self.data_dir, "{}_landmark.txt".format(filename))
            landmark = []
            with open(landmark_path, "r") as text_file:
                for _ in range(5): # 5 Points
                    line = text_file.readline()
                    point = [float(x) for x in line.split()]
                    landmark.append(point[0])
                    landmark.append(point[1])
                landmark = np.array(landmark)

            batch_x.append(img)
            batch_yaw.append([bin_labels[0], cont_labels[0]])
            batch_pitch.append([bin_labels[1], cont_labels[1]])
            batch_roll.append([bin_labels[2], cont_labels[2]])
            batch_landmark.append(landmark)

        batch_x = np.array(batch_x, dtype=np.float32)
        batch_yaw = np.array(batch_yaw)
        batch_pitch = np.array(batch_pitch)
        batch_roll = np.array(batch_roll)
        batch_landmark = np.array(batch_landmark)

        return batch_x, [batch_yaw, batch_pitch, batch_roll, batch_landmark]

    def __get_input_img(self, data_dir, file_name, img_ext='.png', annot_ext='.txt', augment=False):
        img = cv2.imread(os.path.join(data_dir, file_name + '_rgb' + img_ext))

        if augment:
            img = augment_img(img)

            # Uncomment following lines to write out augmented images for debuging
            # cv2.imwrite("aug_" + str(random.randint(0, 50)) + ".png", img)
            # cv2.waitKey(0)

        bbox_path = os.path.join(
            data_dir, "{}_bbox.txt".format(file_name))

        # Load bounding box
        with open(bbox_path, "r") as text_file:
            line = text_file.readline().split(' ')
            bbox = [int(line[0]), int(line[1]), int(line[2]), int(line[3])]
    
        # Loosely crop face
        crop_img = utils.crop_face_loosely(bbox, img, self.input_size)
        crop_img = cv2.resize(crop_img, (self.input_size, self.input_size))
        crop_img = np.asarray(crop_img)
        normed_img = (crop_img - crop_img.mean())/crop_img.std()
        
        return normed_img

    def __get_input_label(self, data_dir, file_name, annot_ext='.txt'):
        # Load pose in degrees
        pose_path = os.path.join(data_dir, file_name + '_pose' + annot_ext)
        pose_annot = open(pose_path, 'r')
        R = []
        for line in pose_annot:
            line = line.strip('\n').split(' ')
            l = []
            if line[0] != '':
                for nb in line:
                    if nb == '':
                        continue
                    l.append(float(nb))
                R.append(l)
        
        R = np.array(R)
        T = R[3, :]
        R = R[:3, :]
        pose_annot.close()
        
        R = np.transpose(R)
        
        roll = -np.arctan2(R[1][0], R[0][0]) * 180 / np.pi
        yaw = -np.arctan2(-R[2][0], np.sqrt(R[2][1] ** 2 + R[2][2] ** 2)) * 180 / np.pi
        pitch = np.arctan2(R[2][1], R[2][2]) * 180 / np.pi
        
        # Bin values
        bins = np.array(range(-99, 99, 3))
        binned_labels = np.digitize([yaw, pitch, roll], bins) - 1
    
        cont_labels = [yaw, pitch, roll]

        return binned_labels, cont_labels


    def __get_list_from_filenames(file_path):
        lines = []
        with open(file_path) as fp:
            content = fp.read()
            lines = content.split("\n")
        for l in lines:
            if len(l) > 14:
                wrapped = textwrap.wrap(l, 14)
                lines += wrapped
        lines = [l for l in lines if l != "" and len(l) <= 14]
        return lines


class Biwi:
    def __init__(self, data_dir, data_file, batch_size=64, input_size=64, train_ratio=0.8, val_ratio=0.15):
        self.data_dir = data_dir
        self.data_file = data_file
        self.batch_size = batch_size
        self.input_size = input_size
        self.train_file = None
        self.test_file = None
        self.__gen_filename_list(os.path.join(self.data_dir, self.data_file))
        self.train_num, self.val_num, self.test_num = self.__gen_train_test_file(os.path.join(self.data_dir, 'train.txt'),
            os.path.join(self.data_dir, 'val.txt'),
            os.path.join(self.data_dir, 'test.txt'), train_ratio=train_ratio, val_ratio=val_ratio)

    def __gen_filename_list(self, filename_list_file):
        if not os.path.exists(filename_list_file):
            with open(filename_list_file, 'w+') as tlf:
                for root, dirs, files in os.walk(self.data_dir):
                    for subdir in dirs:
                        subfiles = os.listdir(os.path.join(self.data_dir, subdir))
                    
                        for f in subfiles:
                            if os.path.splitext(f)[1] == '.png':
                                token = os.path.splitext(f)[0].split('_')
                                filename = token[0] + '_' + token[1]
                                # print(filename)
                                tlf.write(subdir + '/' + filename + '\n')
    
    def __gen_train_test_file(self, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.15):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        return split_samples(os.path.join(self.data_dir, self.data_file), train_file, val_file, test_file, train_ratio=train_ratio, val_ratio=val_ratio)
    
    def train_num(self):
        return self.train_num

    def val_num(self):
        return self.val_num
    
    def test_num(self):
        return self.test_num
    
    def save_test(self, name, save_dir, prediction):
        img_path = os.path.join(self.data_dir, name + '_rgb.png')
        # print(img_path)
    
        cv2_img = cv2.imread(img_path)
        cv2_img = utils.draw_axis(cv2_img, prediction[0], prediction[1], prediction[2], tdx=200, tdy=200,
                            size=100)
        cv2_img = utils.draw_landmark(cv2_img, prediction[3])
        save_path = os.path.join(save_dir, name.split('/')[1] + '.png')
        # print(save_path)
        cv2.imwrite(save_path, cv2_img)
        
    def get_data_generator(self, shuffle=True, partition="train"):
        sample_file = self.train_file
        if partition == "test":
            sample_file = self.test_file
        elif partition == "val":
            sample_file = self.val_file

        if partition == "test":
            shuffle = False
            augment = False
        else:
            shuffle = True
            augment = True
        return BIWIDataSequence(self.data_dir, sample_file, self.batch_size, input_size=self.input_size, shuffle=shuffle, augment=augment)