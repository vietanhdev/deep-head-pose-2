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

EPOCHS=25

class Conv2D_BN_ReLU(tf.keras.Model):
    """Conv2D -> BN -> ReLU"""
    def __init__(self, channel, kernel_size=1, stride=1):
        super(Conv2D_BN_ReLU, self).__init__()

        self.conv = Conv2D(channel, kernel_size, strides=stride,
                            padding="SAME", use_bias=False)
        self.bn = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)
        self.relu = Activation("relu")

    def call(self, inputs, training=True):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        x = self.relu(x)
        return x

class DepthwiseConv2D_BN(tf.keras.Model):
    """DepthwiseConv2D -> BN"""
    def __init__(self, kernel_size=3, stride=1):
        super(DepthwiseConv2D_BN, self).__init__()

        self.dconv = DepthwiseConv2D(kernel_size, strides=stride,
                                     depth_multiplier=1,
                                     padding="SAME", use_bias=False)
        self.bn = BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)

    def call(self, inputs, training=True):
        x = self.dconv(inputs)
        x = self.bn(x, training=training)
        return x


def channle_shuffle(inputs, group):
    """Shuffle the channel
    Args:
        inputs: 4D Tensor
        group: int, number of groups
    Returns:
        Shuffled 4D Tensor
    """
    in_shape = inputs.get_shape().as_list()
    h, w, in_channel = in_shape[1:]
    assert in_channel % group == 0
    l = tf.reshape(inputs, [-1, h, w, in_channel // group, group])
    l = tf.transpose(a=l, perm=[0, 1, 2, 4, 3])
    l = tf.reshape(l, [-1, h, w, in_channel])

    return l
    
class ShufflenetUnit1(tf.keras.Model):
    def __init__(self, out_channel):
        """The unit of shufflenetv2 for stride=1
        Args:
            out_channel: int, number of channels
        """
        super(ShufflenetUnit1, self).__init__()

        assert out_channel % 2 == 0
        self.out_channel = out_channel

        self.conv1_bn_relu = Conv2D_BN_ReLU(out_channel // 2, 1, 1)
        self.dconv_bn = DepthwiseConv2D_BN(3, 1)
        self.conv2_bn_relu = Conv2D_BN_ReLU(out_channel // 2, 1, 1)

    def call(self, inputs, training=False):
        # split the channel
        shortcut, x = tf.split(inputs, 2, axis=3)

        x = self.conv1_bn_relu(x, training=training)
        x = self.dconv_bn(x, training=training)
        x = self.conv2_bn_relu(x, training=training)

        x = tf.concat([shortcut, x], axis=3)
        x = channle_shuffle(x, 2)
        return x


class ShufflenetUnit2(tf.keras.Model):
    """The unit of shufflenetv2 for stride=2"""
    def __init__(self, in_channel, out_channel):
        super(ShufflenetUnit2, self).__init__()

        assert out_channel % 2 == 0
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1_bn_relu = Conv2D_BN_ReLU(out_channel // 2, 1, 1)
        self.dconv_bn = DepthwiseConv2D_BN(3, 2)
        self.conv2_bn_relu = Conv2D_BN_ReLU(out_channel - in_channel, 1, 1)

        # for shortcut
        self.shortcut_dconv_bn = DepthwiseConv2D_BN(3, 2)
        self.shortcut_conv_bn_relu = Conv2D_BN_ReLU(in_channel, 1, 1)

    def call(self, inputs, training=False):
        shortcut, x = inputs, inputs

        x = self.conv1_bn_relu(x, training=training)
        x = self.dconv_bn(x, training=training)
        x = self.conv2_bn_relu(x, training=training)

        shortcut = self.shortcut_dconv_bn(shortcut, training=training)
        shortcut = self.shortcut_conv_bn_relu(shortcut, training=training)

        x = tf.concat([shortcut, x], axis=3)
        x = channle_shuffle(x, 2)
        return x

class ShufflenetStage(tf.keras.Model):
    """The stage of shufflenet"""
    def __init__(self, in_channel, out_channel, num_blocks):
        super(ShufflenetStage, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.ops = []
        for i in range(num_blocks):
            if i == 0:
                op = ShufflenetUnit2(in_channel, out_channel)
            else:
                op = ShufflenetUnit1(out_channel)
            self.ops.append(op)

    def call(self, inputs, training=False):
        x = inputs
        for op in self.ops:
            x = op(x, training=training)
        return x


class ShuffleNetv2(tf.keras.Model):
    """Shufflenetv2"""
    def __init__(self, num_classes, first_channel=24, channels_per_stage=(116, 232, 464)):
        super(ShuffleNetv2, self).__init__()

        self.num_classes = num_classes

        self.conv1_bn_relu = Conv2D_BN_ReLU(first_channel, 3, 2)
        self.pool1 = MaxPool2D(3, strides=2, padding="SAME")
        self.stage2 = ShufflenetStage(first_channel, channels_per_stage[0], 4)
        self.stage3 = ShufflenetStage(channels_per_stage[0], channels_per_stage[1], 8)
        self.stage4 = ShufflenetStage(channels_per_stage[1], channels_per_stage[2], 4)
        self.conv5_bn_relu = Conv2D_BN_ReLU(1024, 1, 1)
        self.gap = GlobalAveragePooling2D()

    def call(self, inputs, training=False):
        x = self.conv1_bn_relu(inputs, training=training)
        x = self.pool1(x)
        x = self.stage2(x, training=training)
        x = self.stage3(x, training=training)
        x = self.stage4(x, training=training)
        x = self.conv5_bn_relu(x, training=training)
        x = self.gap(x)
        return x

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

    def train(self, model_path, max_epoches=EPOCHS, tf_board_log_dir="./logs", load_weight=True):
        
        if load_weight:
            self.model.load_weights(model_path)
        else:
            # Define the callbacks for training
            tb = TensorBoard(log_dir=tf_board_log_dir, write_graph=True)
            mc = ModelCheckpoint(filepath=model_path + "_{epoch:02d}_{val_loss:.2f}.h5", save_weights_only=True, save_format="h5", verbose=2)
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
                                    max_queue_size=32,
                                    workers=16,
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
