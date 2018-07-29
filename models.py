# -*- coding: utf-8 -*-

import numpy as np
#import tensorflow as tf
from keras.models import *
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract
#from keras import backend as K

def DnCNN():
    
    inpt = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model





def DnCNN_C():
    
    inpt = Input(shape=(None,None,3))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(18):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model





# cannot load the weights directly from mat
#m.load_weights('DnCNN-matlab/model/specifics/sigma=50.mat')

# so instead load them manually by copying the conv weights layer by layer 

def DnCNN_pretrained_weights(sigma, savefile=None, verbose=False):
    '''
    Loads the pretrained weights of DnCNN for grayscale images from 
    https://github.com/cszn/DnCNN.git
    
    sigma: is the level of noise in range(10,70,5)
    savefile: is the .hdf5 file to save the model weights 
    returns the DnCNN(1,1) model with 17 layers with the pretrained weights
    '''
    
    if sigma  not in list(range(10,70,5)):
        print ('pretained sigma %d is not available'%sigma)
        return
    
    
    # download the pretained weights
    import os
    import subprocess

    here = os.path.dirname(__file__)
    try:
        os.stat(here+'/DnCNN')
    except OSError:
        print('downloading pretrained models')
        subprocess.run(['git', 'clone',  'https://github.com/cszn/DnCNN.git'],cwd=here)
           

    # read the weights        
    import numpy as np
    from keras.models import save_model
    
    m = DnCNN()

    import hdf5storage
    mat = hdf5storage.loadmat(here+'/DnCNN/model/specifics/sigma=%d.mat'%sigma)

    t=0
    for i in [1] + list(range(3,len(m.layers),3)):
        l = m.layers[i]
        if verbose:
            print(i, l.get_weights()[0].shape, l.get_weights()[1].shape)

        x = mat['net'][0][0][0][t]
        if verbose:
            print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)

        #  l.get_weights()[0] = x[0][1][0,0]
        #  l.get_weights()[1] = x[0][1][0,1] 

        l.set_weights([np.reshape(np.array(x[0][1][0,0]), l.get_weights()[0].shape),  
                       np.reshape(np.array(x[0][1][0,1]), l.get_weights()[1].shape)])

        t+=2

    if savefile is not None:
        save_model(m, savefile)
        #save_model(m, 'dncnnmodel.hdf5')
        
    return m



def DnCNN_C_pretrained_weights(sigma, savefile=None, verbose=False):
    '''
    Loads the pretrained weights of DnCNN for grayscale images from 
    https://github.com/cszn/DnCNN.git
    
    sigma: is the level of noise in [5,10,15,25,35,50]
    savefile: is the .hdf5 file to save the model weights 
    returns the DnCNN(1,1) model with 17 layers with the pretrained weights
    '''
    
    if sigma  not in [5,10,15,25,35,50]:
        print ('pretained sigma %d is not available'%sigma)
        return

    
    # download the pretained weights
    import os
    import subprocess

    here = os.path.dirname(__file__)
    try:
        os.stat(here+'/DnCNN')
    except OSError:
        print('downloading pretrained models')
        subprocess.run(['git', 'clone',  'https://github.com/cszn/DnCNN.git'],cwd=here)
           
    
    # read the weights    
    import numpy as np
    from keras.models import save_model
    
    m = DnCNN_C()

    import hdf5storage
    mat = hdf5storage.loadmat(here+'/DnCNN/model/specifics_color/color_sigma=%d.mat'%sigma)

    t=0
    for i in [1] + list(range(3,len(m.layers),3)):
        l = m.layers[i]
        if verbose:
            print(i, l.get_weights()[0].shape, l.get_weights()[1].shape)

        x = mat['net'][0][0][0][t]
        if verbose:
            print(t, x[0][7], x[0][1][0,0].shape, x[0][1][0,1].shape)

        #  l.get_weights()[0] = x[0][1][0,0]
        #  l.get_weights()[1] = x[0][1][0,1] 

        l.set_weights([np.reshape(np.array(x[0][1][0,0]), l.get_weights()[0].shape),  
                       np.reshape(np.array(x[0][1][0,1]), l.get_weights()[1].shape)])

        t+=2

    if savefile is not None:
        save_model(m, savefile)
        #save_model(m, 'dncnnmodel.hdf5')
        
    return m

