import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from chesapeake_loader import *
from hw7_parser import *


def create_discriminator_histograms(args):
    # Load dataset
    ds_train, _, _, _ = create_datasets(base_dir=args.dataset,
                                        full_sat=False,
                                        patch_size=args.image_size,
                                        fold=args.fold,
                                        cache_dir=args.cache,
                                        repeat_train=False,
                                        shuffle_train=args.shuffle,
                                        batch_size=args.batch,
                                        prefetch=args.prefetch,
                                        num_parallel_calls=args.num_parallel_calls)

    # Load saved models
    d_base = keras.models.load_model(f'results/Base_discriminator')
    g_base = keras.models.load_model(f'results/Base_generator')
    d_alt = keras.models.load_model(f'results/Alt_discriminator')
    g_alt = keras.models.load_model(f'results/Alt_generator')

    pred_real_base = []
    pred_fake_base = []
    pred_fake2_base = []
    pred_real_alt = []
    pred_fake_alt = []
    pred_fake2_alt = []
    for I, L in ds_train.batch(3):
        # Real image/label pairs
        I_real = np.squeeze(I[0, :, :, :, :])
        L_real = np.squeeze(L[0, :, :, :, :])

        # Labels for generated images
        I_fake_no_use = np.squeeze(I[1, :, :, :, :])
        L_fake = np.squeeze(L[1, :, :, :, :])

        # Real images and labels, but decorrelated
        I_fake2 = np.squeeze(I[2, :, :, :, :])
        L_fake2 = np.array((np.squeeze(L[2, :, :, :, :])))
        # Decorrelate the I/L pairs
        np.random.shuffle(L_fake2)

        # Batch size
        nexamples = I_real.shape[0]

        # Produce the inputs to the generator: Labels + noise tensors (smallest to largest)
        inputs_base = [L_fake]
        n_noise_steps = 3
        for i in range(n_noise_steps - 1, -1, -1):
            Z = np.random.normal(size=(nexamples, args.image_size // (2 ** i), args.image_size // (2 ** i), 1))
            inputs_base.append(Z)

        inputs_alt = [L_fake]
        n_noise_steps = 5
        for i in range(n_noise_steps - 1, -1, -1):
            Z = np.random.normal(size=(nexamples, args.image_size // (2 ** i), args.image_size // (2 ** i), 1))
            inputs_alt.append(Z)

        # Generate images given the generator inputs
        I_fake_base = g_base.predict(x=inputs_base)
        I_fake_alt = g_alt.predict(x=inputs_alt)

        pred_real_base.extend(d_base.predict(x=[I_real, L_real]))
        pred_fake_base.extend(d_base.predict(x=[I_fake_base, L_fake]))
        pred_fake2_base.extend(d_base.predict(x=[I_fake2, L_fake2]))

        pred_real_alt.extend(d_alt.predict(x=[I_real, L_real]))
        pred_fake_alt.extend(d_alt.predict(x=[I_fake_alt, L_fake]))
        pred_fake2_alt.extend(d_alt.predict(x=[I_fake2, L_fake2]))

    pred_real_base = np.array(pred_real_base)
    pred_fake_base = np.array(pred_fake_base)
    pred_fake2_base = np.array(pred_fake2_base)
    pred_real_alt = np.array(pred_real_alt)
    pred_fake_alt = np.array(pred_fake_alt)
    pred_fake2_alt = np.array(pred_fake2_alt)

    print(f'shape of pred_real: {pred_real_base.shape}')
    print(f'shape of pred_fake: {pred_fake_base.shape}')
    print(f'shape of pred_fake2: {pred_fake2_base.shape}')


if __name__ == '__main__':
    # Parse and check incoming arguments
    parser = create_parser()
    args = parser.parse_args()
    create_discriminator_histograms(args)
