import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import utils
import math
import time
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, DepthwiseConv2D, GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras import callbacks
from shufflenetv2 import *

class HeadPoseNet:
    def __init__(self, train_dataset=None, val_dataset=None, test_dataset=None, bin_num=66, batch_size=8, input_size=128, learning_rate=0.001):
        self.class_num = bin_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.model = self.__create_model()
        
    def __loss_angle(self, y_true, y_pred, alpha=0.5):
        # cross entropy loss
        bin_true = y_true[:,0]
        cont_true = y_true[:,1]

        # CLS loss
        onehot_labels = tf.one_hot(tf.cast(bin_true, tf.int32), 66)
        onehot_labels = tf.cast(onehot_labels, tf.float32)
        cls_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=y_pred)

        # MSE loss
        pred_cont = tf.reduce_sum(input_tensor=tf.nn.softmax(y_pred) * self.idx_tensor, axis=1) * 3 - 99
        mse_loss = tf.compat.v1.losses.mean_squared_error(labels=cont_true, predictions=pred_cont)

        # Total loss
        total_loss = cls_loss + alpha * mse_loss
        return total_loss

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.input_size, self.input_size, 3))
        feature = ShuffleNetv2(self.class_num)(inputs)
        feature = tf.keras.layers.Flatten()(feature)
        feature = tf.keras.layers.Dropout(0.5)(feature)
        feature = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu)(feature)
        
        fc_yaw = tf.keras.layers.Dense(name='yaw', units=self.class_num)(feature)
        fc_pitch = tf.keras.layers.Dense(name='pitch', units=self.class_num)(feature)
        fc_roll = tf.keras.layers.Dense(name='roll', units=self.class_num)(feature)

        fc_1_landmarks = tf.keras.layers.Dense(512, activation='relu', name='fc_landmarks')(feature)
        fc_2_landmarks = tf.keras.layers.Dense(10, name='landmarks')(fc_1_landmarks)
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll, fc_2_landmarks])
        
        losses = {
            'yaw':self.__loss_angle,
            'pitch':self.__loss_angle,
            'roll':self.__loss_angle,
            'landmarks':'mean_squared_error'
        }
        
        model.compile(optimizer=tf.optimizers.Adam(self.learning_rate),
                        loss=losses, loss_weights=[1, 1, 1, 20000])
       
        return model

    def train(self, model_path, max_epoches=100, tf_board_log_dir="./logs", load_weight=True):
        
        if load_weight:
            self.model.load_weights(model_path)
        else:
            # Define the callbacks for training
            tb = TensorBoard(log_dir=tf_board_log_dir, write_graph=True)
            mc = ModelCheckpoint(filepath=model_path.replace(".h5", "") + "_ep{epoch:03d}.h5", save_weights_only=True, save_format="h5", verbose=2)
            def step_decay(epoch):
                initial_lrate = 0.001
                drop = 0.5
                epochs_drop = 15.0
                lrate = initial_lrate * math.pow(drop,  
                        math.floor((1+epoch)/epochs_drop))
                return lrate
            lr = LearningRateScheduler(step_decay)
            
            self.model.fit_generator(generator=self.train_dataset,
                                    epochs=max_epoches,
                                    steps_per_epoch=len(self.train_dataset),
                                    validation_data=self.val_dataset,
                                    validation_steps=len(self.val_dataset),
                                    max_queue_size=128,
                                    workers=8,
                                    callbacks=[tb, mc, lr],
                                    verbose=1)
            
    def test(self):
        yaw_error = .0
        pitch_error = .0
        roll_error = .0
        landmark_error = .0
        total_time = .0
        total_samples = 0

        test_gen = self.test_dataset
        for images, [batch_yaw, batch_pitch, batch_roll, batch_landmark] in test_gen:

            start_time = time.time()
            batch_yaw_pred, batch_pitch_pred, batch_roll_pred, batch_landmark_pred = self.predict_batch(images, normalize=False)
            total_time += time.time() - start_time
            
            total_samples += np.array(images).shape[0]

            batch_yaw = batch_yaw[:, 1]
            batch_pitch = batch_pitch[:, 1]
            batch_roll = batch_roll[:, 1]
    
            # Mean absolute error
            yaw_error += np.sum(np.abs(batch_yaw - batch_yaw_pred))
            pitch_error += np.sum(np.abs(batch_pitch - batch_pitch_pred))
            roll_error += np.sum(np.abs(batch_roll - batch_roll_pred))
            landmark_error += np.sum(np.abs(batch_landmark - batch_landmark_pred))
        
        avg_time = total_time / total_samples
        avg_fps = 1.0 / avg_time

        print("### MAE: ")
        print("- Yaw MAE: {}".format(yaw_error / len(test_gen)))
        print("- Pitch MAE: {}".format(pitch_error / len(test_gen)))
        print("- Roll MAE: {}".format(roll_error / len(test_gen)))
        print("- Landmark MAE: {}".format(landmark_error / len(test_gen)))
        print("- Avg. FPS: {}".format(avg_fps))
        

    def predict_batch(self, face_imgs, verbose=1, normalize=True):
        if normalize:
            img_batch = self.normalize_img_batch(face_imgs)
        else:
            img_batch = np.array(face_imgs)
        predictions = self.model.predict(img_batch, batch_size=1, verbose=verbose)
        headpose_preds = np.array(predictions[:3], dtype=np.float32)
        pred_cont_yaw = tf.reduce_sum(input_tensor=tf.nn.softmax(headpose_preds[0, :, :]) * self.idx_tensor, axis=1) * 3 - 99
        pred_cont_pitch = tf.reduce_sum(input_tensor=tf.nn.softmax(headpose_preds[1, :, :]) * self.idx_tensor, axis=1) * 3 - 99
        pred_cont_roll = tf.reduce_sum(input_tensor=tf.nn.softmax(headpose_preds[2, :, :]) * self.idx_tensor, axis=1) * 3 - 99
        pred_landmark = predictions[3]
        return pred_cont_yaw, pred_cont_pitch, pred_cont_roll, pred_landmark

    def normalize_img_batch(self, face_imgs):
        image_batch = np.array(face_imgs, dtype=np.float32)
        image_batch = np.asarray(image_batch)
        normed_image_batch = (image_batch - image_batch.mean())/image_batch.std()
        return normed_image_batch
