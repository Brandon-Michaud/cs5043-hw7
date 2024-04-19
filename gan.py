import tensorflow as tf
from keras.layers import Input, Dense, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.models import Model
from network_support import *


def create_discriminator(image_size,
                         n_channels,
                         n_classes,
                         filters,
                         hidden,
                         n_conv_per_step=3,
                         conv_activation='elu',
                         kernel_size=3,
                         padding='valid',
                         sdropout=None,
                         dense_activation='elu',
                         dropout=None,
                         batch_normalization=False,
                         lrate=0.0001,
                         grad_clip=None,
                         loss=None,
                         metrics=None):
    input1 = Input(shape=(image_size[0], image_size[1], n_channels,))
    input2 = Input(shape=(image_size[0], image_size[1], 1,))

    tensor = Concatenate()([input1, input2])

    tensor, _ = create_cnn_down_stack(tensor=tensor,
                                      n_conv_per_step=n_conv_per_step,
                                      filters=filters,
                                      activation=conv_activation,
                                      kernel_size=kernel_size,
                                      padding=padding,
                                      batch_normalization=batch_normalization,
                                      sdropout=sdropout)
    tensor = GlobalMaxPooling2D()(tensor)
    tensor = create_dense_stack(tensor=tensor,
                                nhidden=hidden,
                                activation=dense_activation,
                                batch_normalization=batch_normalization,
                                dropout=dropout)
    tensor = Dense(n_classes, activation='sigmoid')(tensor)
    output = tensor

    # Create model from data flow
    model = Model(inputs=[input1, input2], outputs=output)

    # The optimizer determines how the gradient descent is to be done
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False, clipnorm=grad_clip)

    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


def create_generator():
    print()


def create_gan():
    model = create_discriminator(image_size=[],
                                 n_channels=0,
                                 n_classes=0,
                                 filters=[],
                                 hidden=[],
                                 n_conv_per_step=0,
                                 conv_activation='elu',
                                 kernel_size=3,
                                 padding='elu',
                                 sdropout=None,
                                 dense_activation='elu',
                                 dropout=None,
                                 batch_normalization=False,
                                 lrate=0.0001,
                                 grad_clip=None,
                                 loss=None,
                                 metrics=None)
