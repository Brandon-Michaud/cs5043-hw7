import tensorflow as tf
from keras.layers import Input, Dense, GlobalMaxPooling2D, UpSampling2D, Concatenate, Conv2D
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
                         # lrate=0.0001,
                         # grad_clip=None,
                         # loss=None,
                         # metrics=None
                         ):
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
    # opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False, clipnorm=grad_clip)
    #
    # model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


def create_generator(image_size,
                     n_channels,
                     n_classes,
                     n_noise_steps,
                     filters,
                     n_conv_per_step=3,
                     conv_activation='elu',
                     kernel_size=3,
                     padding='valid',
                     sdropout=None,
                     batch_normalization=False,
                     # lrate=0.0001,
                     # grad_clip=None,
                     # loss=None,
                     # metrics=None
                     ):
    # Input image with labels
    tensor = Input(shape=(image_size[0], image_size[1], n_classes,))
    inputs = [tensor]

    # Input noises
    noises = []
    for i in range(n_noise_steps - 1, -1, -1):
        noise = Input(shape=(image_size[0] // (2 ** i), image_size[1] // (2 ** i), 1,))
        inputs.append(noise)
        noises.append(noise)

    # Down convolutions in Unet
    tensor, skips = create_cnn_down_stack(tensor=tensor,
                                          n_conv_per_step=n_conv_per_step,
                                          filters=filters,
                                          activation=conv_activation,
                                          kernel_size=kernel_size,
                                          padding=padding,
                                          batch_normalization=batch_normalization,
                                          sdropout=sdropout)

    # Get rid of last skip connection (unneeded)
    skips.pop()

    # Up convolutions in Unet
    tensor = Concatenate()([tensor, noises.pop(0)])
    r_filters = list(reversed(filters))

    for i, f in enumerate(r_filters[:-1]):
        # Last element in the previous conv stack, but we increase the number of filters
        tensor = create_conv_stack(tensor=tensor,
                                   n_conv_per_step=n_conv_per_step - 1,
                                   filters=f,
                                   activation=conv_activation,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   batch_normalization=batch_normalization,
                                   sdropout=sdropout)

        # Up Sampling + striding
        tensor = UpSampling2D(size=2)(tensor)

        # Concatenate skip connection and noise
        tensor = Concatenate()([tensor, skips.pop(), noises.pop(0)])

        # Next stack of Conv layers
        tensor = create_conv_stack(tensor=tensor,
                                   n_conv_per_step=1,
                                   filters=f,
                                   activation=conv_activation,
                                   kernel_size=kernel_size,
                                   padding=padding,
                                   batch_normalization=batch_normalization,
                                   sdropout=sdropout)

    # Finish top layer of Unet
    tensor = create_conv_stack(tensor=tensor,
                               n_conv_per_step=n_conv_per_step - 1,
                               filters=r_filters[-1],
                               activation=conv_activation,
                               kernel_size=kernel_size,
                               padding=padding,
                               batch_normalization=batch_normalization,
                               sdropout=sdropout)

    # Add last convolution to output image
    tensor = Conv2D(filters=n_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    use_bias=True,
                    kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='sigmoid')(tensor)

    output = tensor

    # Create model from data flow
    model = Model(inputs=inputs, outputs=output)

    # The optimizer determines how the gradient descent is to be done
    # opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False, clipnorm=grad_clip)
    #
    # model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model


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
    model = create_discriminator(image_size=(args.image_size, args.image_size),
                                 n_channels=3,
                                 n_classes=num_classes,
                                 filters=args.d_filters,
                                 hidden=args.d_hidden,
                                 n_conv_per_step=args.d_n_conv_per_step,
                                 conv_activation=args.d_conv_activation,
                                 kernel_size=args.d_kernel_size,
                                 padding=args.d_padding,
                                 sdropout=args.d_sdropout,
                                 dense_activation=args.d_dense_activation,
                                 dropout=args.d_dropout,
                                 batch_normalization=args.d_batch_normalization,
                                 lrate=args.d_lrate,
                                 grad_clip=args.d_grad_clip,
                                 loss=tf.keras.losses.BinaryCrossentropy,
                                 metrics=[tf.keras.metrics.BinaryAccuracy])
    model = create_generator(image_size=(args.image_size, args.image_size),
                             n_channels=3,
                             n_classes=num_classes,
                             n_noise_steps=args.g_n_noise_steps,
                             filters=args.g_filters,
                             n_conv_per_step=args.g_n_conv_per_step,
                             conv_activation=args.g_conv_activation,
                             kernel_size=args.g_kernel_size,
                             padding=args.g_padding,
                             sdropout=args.g_sdropout,
                             batch_normalization=args.g_batch_normalization)
