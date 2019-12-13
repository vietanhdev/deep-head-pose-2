import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, DepthwiseConv2D, GlobalAveragePooling2D

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