'''
High-level GAN tools

Andrew H. Fagg
Advanced Machine Learning

'''

import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt

def train_loop(g_model:keras.Model, 
               d_model:keras.Model, 
               gtrain_model:keras.Model, 
               ds_train:tf.data.Dataset, 
               n_noise_steps:int,
              image_size:int,
               nepochs_meta:int=10,
              nepochs_d:int=10,
              nepochs_g:int=1,
              verbose:bool=True):
    '''
    Generator / Discriminator co-training
    
    Loop over sets of 3 image set batches to generate the training data:
    - Image set 0: Real I/L pairs
    - Image set 1: Generated I / Real L pairs
    - Image set 2: Real I and L, but they are not paired
    
    :param g_model: Generator model
    :param d_model: Discriminator model
    :param gtrain_model: Model for training the generator
    :param ds_train: Training data set
    :param n_noise_steps: Number of random tensors that the generator is expecting
    :param image_size: Integer: Number of rows/cols in the generated & example images
    :param nepochs_meta: Number of training passes for both models
    :param nepochs_d: Number of training epochs for the discriminator for each pass
    :param nepochs_g: Number of training epochs for the generator for each pass
    
    :return: Last set of fake labels, the corresponding generated images, and the 
                         true images that correspond to the labels
    '''
    
    for I, L in ds_train.batch(3).take(nepochs_meta):
        # Real image/label pairs
        I_real = np.squeeze(I[0,:,:,:,:])
        L_real = np.squeeze(L[0,:,:,:,:])

        # Labels for generated images
        I_fake_no_use = np.squeeze(I[1,:,:,:,:])
        L_fake = np.squeeze(L[1,:,:,:,:])

        # Real images and labels, but decorrelated
        I_fake2 = np.squeeze(I[2,:,:,:,:])
        L_fake2 = np.array((np.squeeze(L[2,:,:,:,:])))
        # Decorrelate the I/L pairs
        np.random.shuffle(L_fake2)

        # Batch size
        nexamples = I_real.shape[0]

        # Produce the inputs to the generator: Labels + noise tensors (smallest to largest)
        inputs =[L_fake]
        for i in range(n_noise_steps-1,-1,-1):
            Z = np.random.normal(size=(nexamples, image_size//(2**i), image_size//(2**i), 1))
            inputs.append(Z)

        # Generate images given the generator inputs
        I_fake = g_model.predict(x=inputs) 

        # Append the real set + two fake sets together
        I_all = np.concatenate([I_real, I_fake, I_fake2], axis=0)
        L_all = np.concatenate([L_real, L_fake, L_fake2], axis=0)

        # Create the labels for the descriminator.  Label '1' for real pairs
        #   and '0' for the others
        desired = np.concatenate([np.ones((nexamples,1)), np.zeros((nexamples*2,1))])

        # Train the discriminator: use the 3 sets
        print('DISCRIMINATOR')
        d_model.fit(x=[I_all, L_all], y = desired, epochs=nepochs_d, verbose=verbose)

        # Train the generator: only use the fake set with the generated images.
        #  Desired output is '1' for every image (we want to fool the descriminator)
        print('GENERATOR')
        gtrain_model.fit(x=inputs, y=np.ones((nexamples,1)), epochs=nepochs_g, verbose=verbose)
    print('DONE')
    
    # Return the last version of labels + generated images
    return L_fake, I_fake, I_fake_no_use
    
def render_examples(L_fake:np.array, 
                    I_fake:np.array, 
                    I_fake_no_use:np.array,
                   n:int=40):
    '''
    Show generated images: one example per row
    
    :param L_fake: Tensor containing a set of 1-hot encoded labeled images
    :param I_fake: Tensor containing a set of generated RGB images
    :param I_fake_no_use: Tensor containing a set of real RGB images
    :param n: number of examples to show
    
    :return: Resulting figure handle
    
    '''
    cmap = colors.ListedColormap(['k','b','y','g','r'])

    fig, axs=plt.subplots(n,3, figsize=(16,300))
    Lc_fake = np.argmax(L_fake, axis=-1)
    for i in range(n):
        if i == 0:
            axs[i,0].set_title('Labels')
            axs[i,1].set_title('Generated Image')
            axs[i,2].set_title('True Image')
        # Labels
        axs[i,0].imshow(Lc_fake[i,:,:], vmin=0, vmax=6)
        axs[i,0].set_xticks([])
        axs[i,0].set_yticks([])
        
        # Fake image
        axs[i,1].imshow(I_fake[i,:,:,:])
        axs[i,1].set_xticks([])
        axs[i,1].set_yticks([])
        
        # Real image (but never used)
        axs[i,2].imshow(I_fake_no_use[i,:,:,:])
        axs[i,2].set_xticks([])
        axs[i,2].set_yticks([])
        
    return fig
