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
from backbonds.shufflenetv2_backbond import *
import efficientnet.tfkeras as efn 
import pathlib

class HeadPoseNet:
    def __init__(self, im_width, im_height, nb_bins=66, learning_rate=0.001, loss_weights=[1,1,1,20000], loss_angle_alpha=0.5, backbond="SHUFFLE_NET_V2"):
        self.im_width = im_width
        self.im_height = im_height
        self.class_num = nb_bins
        self.learning_rate = learning_rate
        self.loss_weights = loss_weights
        self.loss_angle_alpha = loss_angle_alpha
        self.backbond = backbond
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.model = self.__create_model()
        
    def __loss_angle(self, y_true, y_pred):
        # Cross entropy loss
        cont_true = y_true[:,0]
        bin_true = y_true[:,1]

        # CLS loss
        onehot_labels = tf.one_hot(tf.cast(bin_true, tf.int32), 66)
        onehot_labels = tf.cast(onehot_labels, tf.float32)
        cls_loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=y_pred)

        # MSE loss
        pred_cont = tf.reduce_sum(input_tensor=tf.nn.softmax(y_pred) * self.idx_tensor, axis=1) * 3 - 99
        mse_loss = tf.compat.v1.losses.mean_squared_error(labels=cont_true, predictions=pred_cont)

        # Total loss
        total_loss = cls_loss + self.loss_angle_alpha * mse_loss
        return total_loss

    def __create_model(self):
        inputs = tf.keras.layers.Input(shape=(self.im_height, self.im_width, 3))

        if self.backbond == "SHUFFLE_NET_V2":
            feature = ShuffleNetv2(self.class_num)(inputs)
            feature = tf.keras.layers.Flatten()(feature)
        elif self.backbond == "EFFICIENT_NET_B0":
            efn_backbond = efn.EfficientNetB0(weights='imagenet', include_top=False, input_shape=(self.im_height, self.im_width, 3))
            efn_backbond.trainable = False
            feature = efn_backbond(inputs)
            feature = tf.keras.layers.Flatten()(feature)
            feature = tf.keras.layers.Dense(1024, activation='relu')(feature)
        else:
            raise ValueError('No such arch!... Please check the backend in config file')

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
                        loss=losses, loss_weights=self.loss_weights)
       
        return model

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def train(self, train_dataset, val_dataset, train_conf):

        # Load pretrained model
        if train_conf["load_weights"]:
            print("Loading model weights: " + train_conf["pretrained_weights_path"])
            self.model.load_weights(train_conf["pretrained_weights_path"])

        # Make model path
        pathlib.Path(train_conf["model_folder"]).mkdir(parents=True, exist_ok=True)

        # Define the callbacks for training
        tb = TensorBoard(log_dir=train_conf["logs_dir"], write_graph=True)
        mc = ModelCheckpoint(filepath=os.path.join(train_conf["model_folder"], train_conf["model_base_name"] + "_ep{epoch:03d}.h5"), save_weights_only=True, save_format="h5", verbose=2)
        def step_decay(epoch, lr):
            drop = train_conf["learning_rate_drop"]
            epochs_drop = train_conf["learning_rate_epochs_drop"]
            lrate = lr * math.pow(drop,math.floor((1+epoch)/epochs_drop))
            return lrate
        lr = LearningRateScheduler(step_decay)
        
        self.model.fit_generator(generator=train_dataset,
                                epochs=train_conf["nb_epochs"],
                                steps_per_epoch=len(train_dataset),
                                validation_data=val_dataset,
                                validation_steps=len(val_dataset),
                                max_queue_size=64,
                                workers=6,
                                callbacks=[tb, mc, lr],
                                verbose=1)
            
    def test(self, test_dataset, show_result=False):
        yaw_error = .0
        pitch_error = .0
        roll_error = .0
        landmark_error = .0
        total_time = .0
        total_samples = 0

        test_dataset.set_normalization(False)
        for images, [batch_yaw, batch_pitch, batch_roll, batch_landmark] in test_dataset:

            start_time = time.time()
            batch_yaw_pred, batch_pitch_pred, batch_roll_pred, batch_landmark_pred = self.predict_batch(images, normalize=True)
            total_time += time.time() - start_time
            
            total_samples += np.array(images).shape[0]

            batch_yaw = batch_yaw[:, 0]
            batch_pitch = batch_pitch[:, 0]
            batch_roll = batch_roll[:, 0]
    
            # Mean absolute error
            yaw_error += np.sum(np.abs(batch_yaw - batch_yaw_pred))
            pitch_error += np.sum(np.abs(batch_pitch - batch_pitch_pred))
            roll_error += np.sum(np.abs(batch_roll - batch_roll_pred))
            landmark_error += np.sum(np.abs(batch_landmark - batch_landmark_pred))

            # Show result
            if show_result:
                for i in range(images.shape[0]):
                    image = images[i]
                    yaw = batch_yaw_pred[i]
                    pitch = batch_pitch_pred[i]
                    roll = batch_roll_pred[i]
                    landmark = batch_landmark_pred[i]

                    image = utils.draw_landmark(image, landmark)
                    image = utils.plot_pose_cube(image, yaw, pitch, roll, tdx=image.shape[1] // 2, tdy=image.shape[0] // 2, size=80)
                    cv2.imshow("Test result", image)
                    cv2.waitKey(0)
        
        avg_time = total_time / total_samples
        avg_fps = 1.0 / avg_time

        print("### MAE: ")
        print("- Yaw MAE: {}".format(yaw_error / total_samples))
        print("- Pitch MAE: {}".format(pitch_error / total_samples))
        print("- Roll MAE: {}".format(roll_error / total_samples))
        print("- Head pose MAE: {}".format((yaw_error + pitch_error + roll_error) / total_samples / 3))
        print("- Landmark MAE: {}".format(landmark_error / total_samples / 10))
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
        image_batch /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        image_batch[..., 0] -= mean[0]
        image_batch[..., 1] -= mean[1]
        image_batch[..., 2] -= mean[2]
        image_batch[..., 0] /= std[0]
        image_batch[..., 1] /= std[1]
        image_batch[..., 2] /= std[2]
        return image_batch
