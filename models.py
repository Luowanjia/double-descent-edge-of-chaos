import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, Activation, Flatten, Lambda,
    BatchNormalization, MaxPooling2D,
    GlobalAveragePooling2D, Reshape
)
from tensorflow.keras.models import Model


def _get_base_width(args, default=8):
    return int(getattr(args, 'base_width', default))


def _get_num_classes(args, default=10):
    return int(getattr(args, 'num_classes', default))


class ConvBNReLUPoolBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, pool_size, block_name):
        super().__init__(name=block_name)
        self.conv = Conv2D(
            out_channels,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            name=f'{block_name}_conv'
        )
        self.bn = BatchNormalization(name=f'{block_name}_bn')
        self.relu = Activation('relu', name=f'{block_name}_relu')
        self.pool = MaxPooling2D(
            pool_size=(pool_size, pool_size),
            strides=(pool_size, pool_size),
            padding='valid',
            name=f'{block_name}_pool'
        )

    def call(self, x, training=None):
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = self.relu(x)
        x = self.pool(x)
        return x


class PaperCNN(Model):
    """
    Paper-style standard CNN:
      4 x [Conv-BN-ReLU-MaxPool] + FC
    widths = [k, 2k, 4k, 8k]
    pool sizes = [1, 2, 2, 8]
    output = logits
    """
    def __init__(self, input_shape, num_classes=10, base_width=8, **kwargs):
        super().__init__(**kwargs)
        k = int(base_width)

        self.block1 = ConvBNReLUPoolBlock(k, 1, 'block1')
        self.block2 = ConvBNReLUPoolBlock(2 * k, 2, 'block2')
        self.block3 = ConvBNReLUPoolBlock(4 * k, 2, 'block3')
        self.block4 = ConvBNReLUPoolBlock(8 * k, 8, 'block4')

        self.flatten = Flatten(name='flatten')
        self.fc = Dense(num_classes, activation=None, name='logits')

        self._input_spec_shape = input_shape

    def forward_features(self, x, training=None):
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        return x

    def call(self, x, training=None):
        x = self.forward_features(x, training=training)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class PaperCNNDynamics(Model):
    """
    Teacher-compatible CNN dynamics model.

    Key design:
    - shared paper-style CNN backbone
    - BEFORE the final logits/softmax, insert an extra projection layer
      whose output dimension matches the input image dimensions (H, W, C)
    - classification path goes THROUGH this projection layer
    - dynamics path uses that projection layer output directly as x_{t+1}=f(x_t)

    This ensures:
    1) model family stays close to the double-descent CNN literature
    2) the dynamics map is image-shaped
    3) the projection head receives classification gradients even when
       aux_recon_weight = 0
    """
    def __init__(self, input_shape, num_classes=10, base_width=8, **kwargs):
        super().__init__(**kwargs)

        H, W, C = input_shape
        k = int(base_width)

        self.block1 = ConvBNReLUPoolBlock(k, 1, 'block1')
        self.block2 = ConvBNReLUPoolBlock(2 * k, 2, 'block2')
        self.block3 = ConvBNReLUPoolBlock(4 * k, 2, 'block3')
        self.block4 = ConvBNReLUPoolBlock(8 * k, 8, 'block4')

        self.proj_gap = GlobalAveragePooling2D(name='proj_gap')
        self.proj_dense = Dense(H * W * C, activation=None, name='proj_dense')
        self.projection_layer = Reshape((H, W, C), name='projection_layer')

        self.cls_flatten = Flatten(name='cls_flatten')
        self.cls_logits = Dense(num_classes, activation=None, name='logits')

        self._input_spec_shape = input_shape

    def forward_features(self, x, training=None):
        x = self.block1(x, training=training)
        x = self.block2(x, training=training)
        x = self.block3(x, training=training)
        x = self.block4(x, training=training)
        return x

    def forward_dynamics(self, x, training=None):
        h = self.forward_features(x, training=training)
        h = self.proj_gap(h)
        h = self.proj_dense(h)
        x_next = self.projection_layer(h)
        return x_next

    def forward_classification(self, x, training=None):
        x_proj = self.forward_dynamics(x, training=training)
        h = self.cls_flatten(x_proj)
        logits = self.cls_logits(h)
        return logits

    def call(self, x, training=None):
        return self.forward_classification(x, training=training)


def mlp(args):
    activation_func = tf.nn.relu
    model = Sequential([
        Dense(784, activation=activation_func, input_shape=(784,)),
        Dense(100, activation=activation_func),
        Dense(784, activation=activation_func),
        Dense(10, activation=tf.nn.softmax)
    ])
    return model


def cnn_3(args):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 2)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def cnn(args):
    base_width = _get_base_width(args, default=8)
    num_classes = _get_num_classes(args, default=10)
    return PaperCNN(
        input_shape=args.input_shape,
        num_classes=num_classes,
        base_width=base_width,
        name='paper_cnn'
    )


def cnn_7(args):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 8)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def cnn_9(args):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 16)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def cnn_without_pooling(args):
    base_width = _get_base_width(args, default=8)
    k = int(base_width)

    model = Sequential()
    model.add(Conv2D(k, (3, 3), padding='same', input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(2 * k, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(4 * k, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8 * k, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def cnn_with_aug(args):
    base_width = _get_base_width(args, default=8)
    num_classes = _get_num_classes(args, default=10)
    return PaperCNN(
        input_shape=args.input_shape,
        num_classes=num_classes,
        base_width=base_width,
        name='paper_cnn_aug'
    )


def cnn_dropout(args):
    base_width = _get_base_width(args, default=8)
    k = int(base_width)

    model = Sequential()
    model.add(Conv2D(k, (3, 3), padding='same',
                     input_shape=args.input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(2 * k, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(4 * k, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(8 * k, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Lambda(lambda x: tf.nn.depth_to_space(x, 4)))
    model.add(Conv2D(3, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def cnn_dynamics(args):
    base_width = _get_base_width(args, default=8)
    num_classes = _get_num_classes(args, default=10)
    return PaperCNNDynamics(
        input_shape=args.input_shape,
        num_classes=num_classes,
        base_width=base_width,
        name='paper_cnn_dynamics'
    )
