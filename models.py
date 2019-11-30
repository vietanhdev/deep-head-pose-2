import tensorflow as tf
import os
import numpy as np
import cv2
import scipy.io as sio
import utils
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

def mask_to_categorical(image, mask):
    mask = tf.one_hot(tf.cast(mask, tf.int32), 3)
    mask = tf.cast(mask, tf.float32)
    return image, mask
class HeadPoseNet:
    def __init__(self, dataset, class_num, batch_size, input_size):
        self.class_num = class_num
        self.batch_size = batch_size
        self.input_size = input_size
        self.idx_tensor = [idx for idx in range(self.class_num)]
        self.idx_tensor = tf.Variable(np.array(self.idx_tensor, dtype=np.float32))
        self.dataset = dataset
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
    
        model = tf.keras.Model(inputs=inputs, outputs=[fc_yaw, fc_pitch, fc_roll])
        
        losses = {
            'yaw':self.__loss_angle,
            'pitch':self.__loss_angle,
            'roll':self.__loss_angle,
        }
        
        model.compile(optimizer=tf.optimizers.Adam(),
                        loss=losses)
       
        return model

    def train(self, model_path, max_epoches=EPOCHS, tf_board_log_dir="./logs", load_weight=True):
        
        if load_weight:
            self.model.load_weights(model_path)
        else:
            # Define the callbacks for training
            tb = TensorBoard(log_dir=tf_board_log_dir, write_graph=True)
            mc = ModelCheckpoint(filepath=model_path, save_weights_only=False, verbose=2)
            
            self.model.fit_generator(generator=self.dataset.data_generator(partition="train"),
                                    epochs=max_epoches,
                                    steps_per_epoch=self.dataset.train_num // self.batch_size,
                                    validation_data=self.dataset.data_generator(partition="val"),
                                    validation_steps=self.dataset.val_num // self.batch_size,
                                    max_queue_size=10,
                                    workers=1,
                                    callbacks=[tb, mc],
                                    verbose=1)
            
    def test(self, save_dir):
        for i, (images, [batch_yaw, batch_pitch, batch_roll], names) in enumerate(self.dataset.data_generator(partition="test")):
            predictions = self.model.predict(images, batch_size=self.batch_size, verbose=1)
            predictions = np.asarray(predictions)
            pred_cont_yaw = tf.reduce_sum(input_tensor=tf.nn.softmax(predictions[0,:,:]) * self.idx_tensor, axis=1) * 3 - 99
            pred_cont_pitch = tf.reduce_sum(input_tensor=tf.nn.softmax(predictions[1,:,:]) * self.idx_tensor, axis=1) * 3 - 99
            pred_cont_roll = tf.reduce_sum(input_tensor=tf.nn.softmax(predictions[2,:,:]) * self.idx_tensor, axis=1) * 3 - 99
            # print(pred_cont_yaw.shape)
            
            self.dataset.save_test(names[0], save_dir, [pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]])
            
    def test_online(self, face_imgs):
        batch_x = np.array(face_imgs, dtype=np.float32)
        predictions = self.model.predict(batch_x, batch_size=1, verbose=1)
        predictions = np.asarray(predictions)
        # print(predictions)
        pred_cont_yaw = tf.reduce_sum(input_tensor=tf.nn.softmax(predictions[0, :, :]) * self.idx_tensor, axis=1) * 3 - 99
        pred_cont_pitch = tf.reduce_sum(input_tensor=tf.nn.softmax(predictions[1, :, :]) * self.idx_tensor, axis=1) * 3 - 99
        pred_cont_roll = tf.reduce_sum(input_tensor=tf.nn.softmax(predictions[2, :, :]) * self.idx_tensor, axis=1) * 3 - 99
        
        return pred_cont_yaw[0], pred_cont_pitch[0], pred_cont_roll[0]
