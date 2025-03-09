# """
#     Models used in experiments
# """

# from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, concatenate, BatchNormalization, Activation, add
# from keras.models import Model, model_from_json
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint

# def UNet(length, n_channel=1):
#     """
#         Standard U-Net
    
#     Arguments:
#         length {int} -- length of the input signal
    
#     Keyword Arguments:
#         n_channel {int} -- number of channels in the output (default: {1})
    
#     Returns:
#         keras.model -- created model
#     """
    
#     x = 32

#     inputs = Input((length, n_channel))
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling1D(pool_size=2)(conv1)

#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling1D(pool_size=2)(conv2)

#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling1D(pool_size=2)(conv3)

#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
#     conv4 = BatchNormalization()(conv4)
#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
#     conv4 = BatchNormalization()(conv4)
#     pool4 = MaxPooling1D(pool_size=2)(conv4)

#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)

#     up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
#     conv6 = BatchNormalization()(conv6)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
#     conv6 = BatchNormalization()(conv6)

#     up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
#     conv7 = BatchNormalization()(conv7)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
#     conv7 = BatchNormalization()(conv7)

#     up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
#     conv8 = BatchNormalization()(conv8)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
#     conv8 = BatchNormalization()(conv8)

#     up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
#     conv9 = BatchNormalization()(conv9)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
#     conv9 = BatchNormalization()(conv9)

#     conv10 = Conv1D(1, 1)(conv9)

#     model = Model(inputs=[inputs], outputs=[conv10]) 

#     return model


# def UNetWide64(length, n_channel=1):
#     """
#        Wider U-Net with kernels multiples of 64
    
#     Arguments:
#         length {int} -- length of the input signal
    
#     Keyword Arguments:
#         n_channel {int} -- number of channels in the output (default: {1})
    
#     Returns:
#         keras.model -- created model
#     """
    
#     x = 64

#     inputs = Input((length, n_channel))
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling1D(pool_size=2)(conv1)

#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling1D(pool_size=2)(conv2)

#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling1D(pool_size=2)(conv3)

#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
#     conv4 = BatchNormalization()(conv4)
#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
#     conv4 = BatchNormalization()(conv4)
#     pool4 = MaxPooling1D(pool_size=2)(conv4)

#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)

#     up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
#     conv6 = BatchNormalization()(conv6)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
#     conv6 = BatchNormalization()(conv6)

#     up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
#     conv7 = BatchNormalization()(conv7)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
#     conv7 = BatchNormalization()(conv7)

#     up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
#     conv8 = BatchNormalization()(conv8)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
#     conv8 = BatchNormalization()(conv8)

#     up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
#     conv9 = BatchNormalization()(conv9)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
#     conv9 = BatchNormalization()(conv9)

#     conv10 = Conv1D(1, 1)(conv9)

#     model = Model(inputs=[inputs], outputs=[conv10])

    

#     return model


# def UNetDS64(length, n_channel=1):
#     """
#         Deeply supervised U-Net with kernels multiples of 64
    
#     Arguments:
#         length {int} -- length of the input signal
    
#     Keyword Arguments:
#         n_channel {int} -- number of channels in the output (default: {1})
    
#     Returns:
#         keras.model -- created model
#     """
    
#     x = 64

#     inputs = Input((length, n_channel))
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling1D(pool_size=2)(conv1)

#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling1D(pool_size=2)(conv2)

#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling1D(pool_size=2)(conv3)

#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
#     conv4 = BatchNormalization()(conv4)
#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
#     conv4 = BatchNormalization()(conv4)
#     pool4 = MaxPooling1D(pool_size=2)(conv4)

#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)
    
#     level4 = Conv1D(1, 1, name="level4")(conv5)

#     up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
#     conv6 = BatchNormalization()(conv6)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
#     conv6 = BatchNormalization()(conv6)
    
#     level3 = Conv1D(1, 1, name="level3")(conv6)

#     up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
#     conv7 = BatchNormalization()(conv7)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
#     conv7 = BatchNormalization()(conv7)
    
#     level2 = Conv1D(1, 1, name="level2")(conv7)

#     up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
#     conv8 = BatchNormalization()(conv8)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
#     conv8 = BatchNormalization()(conv8)
    
#     level1 = Conv1D(1, 1, name="level1")(conv8)

#     up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
#     conv9 = BatchNormalization()(conv9)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
#     conv9 = BatchNormalization()(conv9)

#     out = Conv1D(1, 1, name="out")(conv9)

#     model = Model(inputs=[inputs], outputs=[out, level1, level2, level3, level4])
    
    

#     return model


# def UNetWide40(length, n_channel=1):
#     """
#        Wider U-Net with kernels multiples of 40
    
#     Arguments:
#         length {int} -- length of the input signal
    
#     Keyword Arguments:
#         n_channel {int} -- number of channels in the output (default: {1})
    
#     Returns:
#         keras.model -- created model
#     """
    
#     x = 40

#     inputs = Input((length, n_channel))
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling1D(pool_size=2)(conv1)

#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling1D(pool_size=2)(conv2)

#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling1D(pool_size=2)(conv3)

#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
#     conv4 = BatchNormalization()(conv4)
#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
#     conv4 = BatchNormalization()(conv4)
#     pool4 = MaxPooling1D(pool_size=2)(conv4)

#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)

#     up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
#     conv6 = BatchNormalization()(conv6)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
#     conv6 = BatchNormalization()(conv6)

#     up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
#     conv7 = BatchNormalization()(conv7)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
#     conv7 = BatchNormalization()(conv7)

#     up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
#     conv8 = BatchNormalization()(conv8)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
#     conv8 = BatchNormalization()(conv8)

#     up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
#     conv9 = BatchNormalization()(conv9)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
#     conv9 = BatchNormalization()(conv9)

#     conv10 = Conv1D(1, 1)(conv9)

#     model = Model(inputs=[inputs], outputs=[conv10])

    

#     return model


# def UNetWide48(length, n_channel=1):
#     """
#        Wider U-Net with kernels multiples of 48
    
#     Arguments:
#         length {int} -- length of the input signal
    
#     Keyword Arguments:
#         n_channel {int} -- number of channels in the output (default: {1})
    
#     Returns:
#         keras.model -- created model
#     """
    
#     x = 48

#     inputs = Input((length, n_channel))
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Conv1D(x,3, activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling1D(pool_size=2)(conv1)

#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(pool1)
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Conv1D(x*2,3, activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling1D(pool_size=2)(conv2)

#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(pool2)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Conv1D(x*4,3, activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling1D(pool_size=2)(conv3)

#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(pool3)
#     conv4 = BatchNormalization()(conv4)
#     conv4 = Conv1D(x*8,3, activation='relu', padding='same')(conv4)
#     conv4 = BatchNormalization()(conv4)
#     pool4 = MaxPooling1D(pool_size=2)(conv4)

#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(pool4)
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Conv1D(x*16, 3, activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)

#     up6 = concatenate([UpSampling1D(size=2)(conv5), conv4], axis=2)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(up6)
#     conv6 = BatchNormalization()(conv6)
#     conv6 = Conv1D(x*8, 3, activation='relu', padding='same')(conv6)
#     conv6 = BatchNormalization()(conv6)

#     up7 = concatenate([UpSampling1D(size=2)(conv6), conv3], axis=2)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(up7)
#     conv7 = BatchNormalization()(conv7)
#     conv7 = Conv1D(x*4, 3, activation='relu', padding='same')(conv7)
#     conv7 = BatchNormalization()(conv7)

#     up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(up8)
#     conv8 = BatchNormalization()(conv8)
#     conv8 = Conv1D(x*2, 3, activation='relu', padding='same')(conv8)
#     conv8 = BatchNormalization()(conv8)

#     up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(up9)
#     conv9 = BatchNormalization()(conv9)
#     conv9 = Conv1D(x, 3, activation='relu', padding='same')(conv9)
#     conv9 = BatchNormalization()(conv9)

#     conv10 = Conv1D(1, 1)(conv9)

#     model = Model(inputs=[inputs], outputs=[conv10])

    

#     return model


# def UNetLite(length, n_channel=1):
#     """
#        Shallower U-Net
    
#     Arguments:
#         length {int} -- length of the input signal
    
#     Keyword Arguments:
#         n_channel {int} -- number of channels in the output (default: {1})
    
#     Returns:
#         keras.model -- created model
#     """

#     inputs = Input((length, n_channel))
#     conv1 = Conv1D(32,3, activation='relu', padding='same')(inputs)
#     conv1 = BatchNormalization()(conv1)
#     conv1 = Conv1D(32,3, activation='relu', padding='same')(conv1)
#     conv1 = BatchNormalization()(conv1)
#     pool1 = MaxPooling1D(pool_size=2)(conv1)

#     conv2 = Conv1D(64,3, activation='relu', padding='same')(pool1)
#     conv2 = BatchNormalization()(conv2)
#     conv2 = Conv1D(64,3, activation='relu', padding='same')(conv2)
#     conv2 = BatchNormalization()(conv2)
#     pool2 = MaxPooling1D(pool_size=2)(conv2)

#     conv3 = Conv1D(128,3, activation='relu', padding='same')(pool2)
#     conv3 = BatchNormalization()(conv3)
#     conv3 = Conv1D(128,3, activation='relu', padding='same')(conv3)
#     conv3 = BatchNormalization()(conv3)
#     pool3 = MaxPooling1D(pool_size=2)(conv3)    

#     conv5 = Conv1D(256, 3, activation='relu', padding='same')(pool3)
#     conv5 = BatchNormalization()(conv5)
#     conv5 = Conv1D(256, 3, activation='relu', padding='same')(conv5)
#     conv5 = BatchNormalization()(conv5)

    
#     up7 = concatenate([UpSampling1D(size=2)(conv5), conv3], axis=2)
#     conv7 = Conv1D(128, 3, activation='relu', padding='same')(up7)
#     conv7 = BatchNormalization()(conv7)
#     conv7 = Conv1D(128, 3, activation='relu', padding='same')(conv7)
#     conv7 = BatchNormalization()(conv7)

#     up8 = concatenate([UpSampling1D(size=2)(conv7), conv2], axis=2)
#     conv8 = Conv1D(64, 3, activation='relu', padding='same')(up8)
#     conv8 = BatchNormalization()(conv8)
#     conv8 = Conv1D(64, 3, activation='relu', padding='same')(conv8)
#     conv8 = BatchNormalization()(conv8)

#     up9 = concatenate([UpSampling1D(size=2)(conv8), conv1], axis=2)
#     conv9 = Conv1D(32, 3, activation='relu', padding='same')(up9)
#     conv9 = BatchNormalization()(conv9)
#     conv9 = Conv1D(32, 3, activation='relu', padding='same')(conv9)
#     conv9 = BatchNormalization()(conv9)

#     conv10 = Conv1D(1, 1)(conv9)

#     model = Model(inputs=[inputs], outputs=[conv10])

    

#     return model


# def MultiResUNet1D(length, n_channel=1):
#     """
#        1D MultiResUNet
    
#     Arguments:
#         length {int} -- length of the input signal
    
#     Keyword Arguments:
#         n_channel {int} -- number of channels in the output (default: {1})
    
#     Returns:
#         keras.model -- created model
#     """

#     def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
        
#         kernel = 3

#         x = Conv1D(filters, kernel,  padding=padding)(x)
#         x = BatchNormalization()(x)

#         if(activation == None):
#             return x

#         x = Activation(activation, name=name)(x)
#         return x


#     def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
 
#         x = UpSampling1D(size=2)(x)        
#         x = BatchNormalization()(x)
        
#         return x


#     def MultiResBlock(U, inp, alpha = 2.5):
#         '''
#         MultiRes Block
        
#         Arguments:
#             U {int} -- Number of filters in a corrsponding UNet stage
#             inp {keras layer} -- input layer 
        
#         Returns:
#             [keras layer] -- [output layer]
#         '''

#         W = alpha * U

#         shortcut = inp

#         shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
#                             int(W*0.5), 1, 1, activation=None, padding='same')

#         conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
#                             activation='relu', padding='same')

#         conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
#                             activation='relu', padding='same')

#         conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
#                             activation='relu', padding='same')

#         out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
#         out = BatchNormalization()(out)

#         out = add([shortcut, out])
#         out = Activation('relu')(out)
#         out = BatchNormalization()(out)

#         return out


#     def ResPath(filters, length, inp):
#         '''
#         ResPath
        
#         Arguments:
#             filters {int} -- [description]
#             length {int} -- length of ResPath
#             inp {keras layer} -- input layer 
        
#         Returns:
#             [keras layer] -- [output layer]
#         '''


#         shortcut = inp
#         shortcut = conv2d_bn(shortcut, filters, 1, 1,
#                             activation=None, padding='same')

#         out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

#         out = add([shortcut, out])
#         out = Activation('relu')(out)
#         out = BatchNormalization()(out)

#         for i in range(length-1):

#             shortcut = out
#             shortcut = conv2d_bn(shortcut, filters, 1, 1,
#                                 activation=None, padding='same')

#             out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

#             out = add([shortcut, out])
#             out = Activation('relu')(out)
#             out = BatchNormalization()(out)

#         return out





#     inputs = Input((length, n_channel))

#     mresblock1 = MultiResBlock(32, inputs)
#     pool1 = MaxPooling1D(pool_size=2)(mresblock1)
#     mresblock1 = ResPath(32, 4, mresblock1)

#     mresblock2 = MultiResBlock(32*2, pool1)
#     pool2 = MaxPooling1D(pool_size=2)(mresblock2)
#     mresblock2 = ResPath(32*2, 3, mresblock2)

#     mresblock3 = MultiResBlock(32*4, pool2)
#     pool3 = MaxPooling1D(pool_size=2)(mresblock3)
#     mresblock3 = ResPath(32*4, 2, mresblock3)

#     mresblock4 = MultiResBlock(32*8, pool3)
#     pool4 = MaxPooling1D(pool_size=2)(mresblock4)
#     mresblock4 = ResPath(32*8, 1, mresblock4)

#     mresblock5 = MultiResBlock(32*16, pool4)

#     up6 = concatenate([UpSampling1D(size=2)(mresblock5), mresblock4], axis=-1)
#     mresblock6 = MultiResBlock(32*8, up6)

#     up7 = concatenate([UpSampling1D(size=2)(mresblock6), mresblock3], axis=-1)
#     mresblock7 = MultiResBlock(32*4, up7)

#     up8 = concatenate([UpSampling1D(size=2)(mresblock7), mresblock2], axis=-1)
#     mresblock8 = MultiResBlock(32*2, up8)

#     up9 = concatenate([UpSampling1D(size=2)(mresblock8), mresblock1], axis=-1)
#     mresblock9 = MultiResBlock(32, up9)

#     conv10 = Conv1D(1, 1)(mresblock9)
    
#     model = Model(inputs=[inputs], outputs=[conv10])

#     return model


# def MultiResUNetDS(length, n_channel=1):
#     """
#        1D Deeply Supervised MultiResUNet
    
#     Arguments:
#         length {int} -- length of the input signal
    
#     Keyword Arguments:
#         n_channel {int} -- number of channels in the output (default: {1})
    
#     Returns:
#         keras.model -- created model
#     """

#     def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
        
#         kernel = 3

#         x = Conv1D(filters, kernel,  padding=padding)(x)
#         x = BatchNormalization()(x)

#         if(activation == None):
#             return x

#         x = Activation(activation, name=name)(x)
#         return x


#     def trans_conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(2, 2), name=None):
 
#         x = UpSampling1D(size=2)(x)        
#         x = BatchNormalization()(x)
        
#         return x


#     def MultiResBlock(U, inp, alpha = 2.5):
#         '''
#         MultiRes Block
        
#         Arguments:
#             U {int} -- Number of filters in a corrsponding UNet stage
#             inp {keras layer} -- input layer 
        
#         Returns:
#             [keras layer] -- [output layer]
#         '''

#         W = alpha * U

#         shortcut = inp

#         shortcut = conv2d_bn(shortcut, int(W*0.167) + int(W*0.333) +
#                             int(W*0.5), 1, 1, activation=None, padding='same')

#         conv3x3 = conv2d_bn(inp, int(W*0.167), 3, 3,
#                             activation='relu', padding='same')

#         conv5x5 = conv2d_bn(conv3x3, int(W*0.333), 3, 3,
#                             activation='relu', padding='same')

#         conv7x7 = conv2d_bn(conv5x5, int(W*0.5), 3, 3,
#                             activation='relu', padding='same')

#         out = concatenate([conv3x3, conv5x5, conv7x7], axis=-1)
#         out = BatchNormalization()(out)

#         out = add([shortcut, out])
#         out = Activation('relu')(out)
#         out = BatchNormalization()(out)

#         return out


#     def ResPath(filters, length, inp):
#         '''
#         ResPath
        
#         Arguments:
#             filters {int} -- [description]
#             length {int} -- length of ResPath
#             inp {keras layer} -- input layer 
        
#         Returns:
#             [keras layer] -- [output layer]
#         '''


#         shortcut = inp
#         shortcut = conv2d_bn(shortcut, filters, 1, 1,
#                             activation=None, padding='same')

#         out = conv2d_bn(inp, filters, 3, 3, activation='relu', padding='same')

#         out = add([shortcut, out])
#         out = Activation('relu')(out)
#         out = BatchNormalization()(out)

#         for i in range(length-1):

#             shortcut = out
#             shortcut = conv2d_bn(shortcut, filters, 1, 1,
#                                 activation=None, padding='same')

#             out = conv2d_bn(out, filters, 3, 3, activation='relu', padding='same')

#             out = add([shortcut, out])
#             out = Activation('relu')(out)
#             out = BatchNormalization()(out)

#         return out


#     inputs = Input((length, n_channel))

#     mresblock1 = MultiResBlock(32, inputs)
#     pool1 = MaxPooling1D(pool_size=2)(mresblock1)
#     mresblock1 = ResPath(32, 4, mresblock1)

#     mresblock2 = MultiResBlock(32*2, pool1)
#     pool2 = MaxPooling1D(pool_size=2)(mresblock2)
#     mresblock2 = ResPath(32*2, 3, mresblock2)

#     mresblock3 = MultiResBlock(32*4, pool2)
#     pool3 = MaxPooling1D(pool_size=2)(mresblock3)
#     mresblock3 = ResPath(32*4, 2, mresblock3)

#     mresblock4 = MultiResBlock(32*8, pool3)
#     pool4 = MaxPooling1D(pool_size=2)(mresblock4)
#     mresblock4 = ResPath(32*8, 1, mresblock4)

#     mresblock5 = MultiResBlock(32*16, pool4)

#     level4 = Conv1D(1, 1, name="level4")(mresblock5)

#     up6 = concatenate([UpSampling1D(size=2)(mresblock5), mresblock4], axis=-1)
#     mresblock6 = MultiResBlock(32*8, up6)

#     level3 = Conv1D(1, 1, name="level3")(mresblock6)

#     up7 = concatenate([UpSampling1D(size=2)(mresblock6), mresblock3], axis=-1)
#     mresblock7 = MultiResBlock(32*4, up7)

#     level2 = Conv1D(1, 1, name="level2")(mresblock7)

#     up8 = concatenate([UpSampling1D(size=2)(mresblock7), mresblock2], axis=-1)
#     mresblock8 = MultiResBlock(32*2, up8)
    
#     level1 = Conv1D(1, 1, name="level1")(mresblock8)

#     up9 = concatenate([UpSampling1D(size=2)(mresblock8), mresblock1], axis=-1)
#     mresblock9 = MultiResBlock(32, up9)

#     out = Conv1D(1, 1, name="out")(mresblock9)
    
#     model = Model(inputs=[inputs], outputs=[out,level1,level2,level3,level4])

#     return model
    









###########################################

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPRegression(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegression, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # 输出2个值 [SBP, DBP]
        )

    def forward(self, x):
        return self.fc(x)


class ConvBlock(nn.Module):
    """Convolutional Block with Batch Normalization and ReLU Activation"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class UNet1D(nn.Module):
    """U-Net 1D Model"""
    def __init__(self, length, n_channel=1, base_filters=32):
        super(UNet1D, self).__init__()
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Down-sampling
        self.conv1 = ConvBlock(n_channel, base_filters)
        self.conv2 = ConvBlock(base_filters, base_filters * 2)
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.conv5 = ConvBlock(base_filters * 8, base_filters * 16)

        # Up-sampling
        self.up6 = ConvBlock(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up7 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up8 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up9 = ConvBlock(base_filters * 2 + base_filters, base_filters)

        # Output Layer
        self.final_conv = nn.Conv1d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        # Decoder
        up6 = self.upsample(c5)
        up6 = torch.cat([up6, c4], dim=1)
        up6 = self.up6(up6)

        up7 = self.upsample(up6)
        up7 = torch.cat([up7, c3], dim=1)
        up7 = self.up7(up7)

        up8 = self.upsample(up7)
        up8 = torch.cat([up8, c2], dim=1)
        up8 = self.up8(up8)

        up9 = self.upsample(up8)
        up9 = torch.cat([up9, c1], dim=1)
        up9 = self.up9(up9)

        output = self.final_conv(up9)
        return output

class UNetWide64(nn.Module):
    """U-Net 1D Model with Wider Filters"""
    def __init__(self, length, n_channel=1, base_filters=64):
        super(UNetWide64, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Down-sampling
        self.conv1 = ConvBlock(n_channel, base_filters)
        self.conv2 = ConvBlock(base_filters, base_filters * 2)
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.conv5 = ConvBlock(base_filters * 8, base_filters * 16)

        # Up-sampling
        self.up6 = ConvBlock(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up7 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up8 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up9 = ConvBlock(base_filters * 2 + base_filters, base_filters)

        # Output Layer
        self.final_conv = nn.Conv1d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        # Decoder
        up6 = self.upsample(c5)
        up6 = torch.cat([up6, c4], dim=1)
        up6 = self.up6(up6)

        up7 = self.upsample(up6)
        up7 = torch.cat([up7, c3], dim=1)
        up7 = self.up7(up7)

        up8 = self.upsample(up7)
        up8 = torch.cat([up8, c2], dim=1)
        up8 = self.up8(up8)

        up9 = self.upsample(up8)
        up9 = torch.cat([up9, c1], dim=1)
        up9 = self.up9(up9)

        output = self.final_conv(up9)
        return output


class UNetDS64(nn.Module):
    """Deeply Supervised U-Net 1D"""
    def __init__(self, length, n_channel=1, base_filters=64):
        super(UNetDS64, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Encoder
        self.conv1 = ConvBlock(n_channel, base_filters)
        self.conv2 = ConvBlock(base_filters, base_filters * 2)
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.conv5 = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.up6 = ConvBlock(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up7 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up8 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up9 = ConvBlock(base_filters * 2 + base_filters, base_filters)

        # Deep Supervision Layers
        self.level4 = nn.Conv1d(base_filters * 16, 1, kernel_size=1)
        self.level3 = nn.Conv1d(base_filters * 8, 1, kernel_size=1)
        self.level2 = nn.Conv1d(base_filters * 4, 1, kernel_size=1)
        self.level1 = nn.Conv1d(base_filters * 2, 1, kernel_size=1)

        # Final Output
        self.final_conv = nn.Conv1d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        level4 = self.level4(c5)

        # Decoder
        up6 = self.upsample(c5)
        up6 = torch.cat([up6, c4], dim=1)
        up6 = self.up6(up6)

        level3 = self.level3(up6)

        up7 = self.upsample(up6)
        up7 = torch.cat([up7, c3], dim=1)
        up7 = self.up7(up7)

        level2 = self.level2(up7)

        up8 = self.upsample(up7)
        up8 = torch.cat([up8, c2], dim=1)
        up8 = self.up8(up8)

        level1 = self.level1(up8)

        up9 = self.upsample(up8)
        up9 = torch.cat([up9, c1], dim=1)
        up9 = self.up9(up9)

        out = self.final_conv(up9)

        return {
            'out': out.permute(0, 2, 1),
            'level1': level1.permute(0, 2, 1),
            'level2': level2.permute(0, 2, 1),
            'level3': level3.permute(0, 2, 1),
            'level4': level4.permute(0, 2, 1)
        }


class UNetWide40(nn.Module):
    """Wider U-Net 1D with base filters = 40"""
    def __init__(self, length, n_channel=1, base_filters=40):
        super(UNetWide40, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Encoder
        self.conv1 = ConvBlock(n_channel, base_filters)
        self.conv2 = ConvBlock(base_filters, base_filters * 2)
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.conv5 = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.up6 = ConvBlock(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up7 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up8 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up9 = ConvBlock(base_filters * 2 + base_filters, base_filters)

        # Output Layer
        self.final_conv = nn.Conv1d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        # Decoder
        up6 = self.upsample(c5)
        up6 = torch.cat([up6, c4], dim=1)
        up6 = self.up6(up6)

        up7 = self.upsample(up6)
        up7 = torch.cat([up7, c3], dim=1)
        up7 = self.up7(up7)

        up8 = self.upsample(up7)
        up8 = torch.cat([up8, c2], dim=1)
        up8 = self.up8(up8)

        up9 = self.upsample(up8)
        up9 = torch.cat([up9, c1], dim=1)
        up9 = self.up9(up9)

        output = self.final_conv(up9)

        return output


class UNetWide48(nn.Module):
    """Wider U-Net 1D with base filters = 48"""
    def __init__(self, length, n_channel=1, base_filters=48):
        super(UNetWide48, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Encoder
        self.conv1 = ConvBlock(n_channel, base_filters)
        self.conv2 = ConvBlock(base_filters, base_filters * 2)
        self.conv3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.conv4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.conv5 = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder
        self.up6 = ConvBlock(base_filters * 16 + base_filters * 8, base_filters * 8)
        self.up7 = ConvBlock(base_filters * 8 + base_filters * 4, base_filters * 4)
        self.up8 = ConvBlock(base_filters * 4 + base_filters * 2, base_filters * 2)
        self.up9 = ConvBlock(base_filters * 2 + base_filters, base_filters)

        # Output Layer
        self.final_conv = nn.Conv1d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c4 = self.conv4(p3)
        p4 = self.pool(c4)

        c5 = self.conv5(p4)

        # Decoder
        up6 = self.upsample(c5)
        up6 = torch.cat([up6, c4], dim=1)
        up6 = self.up6(up6)

        up7 = self.upsample(up6)
        up7 = torch.cat([up7, c3], dim=1)
        up7 = self.up7(up7)

        up8 = self.upsample(up7)
        up8 = torch.cat([up8, c2], dim=1)
        up8 = self.up8(up8)

        up9 = self.upsample(up8)
        up9 = torch.cat([up9, c1], dim=1)
        up9 = self.up9(up9)

        output = self.final_conv(up9)

        return output

class UNetLite(nn.Module):
    """Shallower U-Net 1D"""
    def __init__(self, length, n_channel=1):
        super(UNetLite, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Encoder
        self.conv1 = ConvBlock(n_channel, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv5 = ConvBlock(128, 256)

        # Decoder
        self.up7 = ConvBlock(256 + 128, 128)
        self.up8 = ConvBlock(128 + 64, 64)
        self.up9 = ConvBlock(64 + 32, 32)

        # Output Layer
        self.final_conv = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.conv1(x)
        p1 = self.pool(c1)

        c2 = self.conv2(p1)
        p2 = self.pool(c2)

        c3 = self.conv3(p2)
        p3 = self.pool(c3)

        c5 = self.conv5(p3)

        # Decoder
        up7 = self.upsample(c5)
        up7 = torch.cat([up7, c3], dim=1)
        up7 = self.up7(up7)

        up8 = self.upsample(up7)
        up8 = torch.cat([up8, c2], dim=1)
        up8 = self.up8(up8)

        up9 = self.upsample(up8)
        up9 = torch.cat([up9, c1], dim=1)
        up9 = self.up9(up9)

        output = self.final_conv(up9)

        return output


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, activation=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = activation
        if self.activation:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x

class MultiResBlock(nn.Module):
    def __init__(self, U, in_channels, alpha=2.5):
        super().__init__()
        W = alpha * U
        self.conv3x3 = ConvBNReLU(in_channels, int(W*0.167), 3, padding=1)
        self.conv5x5 = ConvBNReLU(int(W*0.167), int(W*0.333), 3, padding=1)
        self.conv7x7 = ConvBNReLU(int(W*0.333), int(W*0.5), 3, padding=1)
        
        shortcut_out = int(W*0.167) + int(W*0.333) + int(W*0.5)
        self.shortcut = ConvBNReLU(in_channels, shortcut_out, 1, padding=0, activation=False)
        
        self.bn1 = nn.BatchNorm1d(shortcut_out)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(shortcut_out)

    def forward(self, x):
        shortcut = self.shortcut(x)
        conv3 = self.conv3x3(x)
        conv5 = self.conv5x5(conv3)
        conv7 = self.conv7x7(conv5)
        
        out = torch.cat([conv3, conv5, conv7], dim=1)
        out = self.bn1(out)
        out = out + shortcut
        out = self.relu(out)
        out = self.bn2(out)
        return out

class ResPath(nn.Module):
    def __init__(self, filters, length, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(ResPathBlock(in_channels, filters))
        for _ in range(length-1):
            self.blocks.append(ResPathBlock(filters, filters))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ResPathBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBNReLU(in_channels, out_channels, 3, padding=1)
        self.shortcut = ConvBNReLU(in_channels, out_channels, 1, padding=0, activation=False)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv(x)
        out = out + shortcut
        out = self.relu(out)
        out = self.bn(out)
        return out

class MultiResUNet1D(nn.Module):
    def __init__(self, length, n_channel=1):
        super().__init__()
        # Encoder
        self.mres1 = MultiResBlock(32, n_channel)
        self.pool1 = nn.MaxPool1d(2)
        self.respath1 = ResPath(32, 4, self.mres1.conv3x3.conv.out_channels + 
                                self.mres1.conv5x5.conv.out_channels + 
                                self.mres1.conv7x7.conv.out_channels)
        
        self.mres2 = MultiResBlock(64, self.mres1.shortcut.conv.out_channels)
        self.pool2 = nn.MaxPool1d(2)
        self.respath2 = ResPath(64, 3, self.mres2.shortcut.conv.out_channels)
        
        self.mres3 = MultiResBlock(128, self.mres2.shortcut.conv.out_channels)
        self.pool3 = nn.MaxPool1d(2)
        self.respath3 = ResPath(128, 2, self.mres3.shortcut.conv.out_channels)
        
        self.mres4 = MultiResBlock(256, self.mres3.shortcut.conv.out_channels)
        self.pool4 = nn.MaxPool1d(2)
        self.respath4 = ResPath(256, 1, self.mres4.shortcut.conv.out_channels)
        
        self.mres5 = MultiResBlock(512, self.mres4.shortcut.conv.out_channels)
        
        # Decoder
        self.up6 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.mres6 = MultiResBlock(256, self.mres5.shortcut.conv.out_channels + 256)
        
        self.up7 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.mres7 = MultiResBlock(128, self.mres6.shortcut.conv.out_channels + 128)
        
        self.up8 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.mres8 = MultiResBlock(64, self.mres7.shortcut.conv.out_channels + 64)
        
        self.up9 = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.mres9 = MultiResBlock(32, self.mres8.shortcut.conv.out_channels + 32)
        
        self.final_conv = nn.Conv1d(self.mres9.shortcut.conv.out_channels, 1, kernel_size=1)

    def forward(self, x):
        # Input shape: (batch_size, length, channels)
        # x = x.permute(0, 2, 1)
        
        # Encoder
        x1 = self.mres1(x)
        p1 = self.pool1(x1)
        r1 = self.respath1(x1)
        
        x2 = self.mres2(p1)
        p2 = self.pool2(x2)
        r2 = self.respath2(x2)
        
        x3 = self.mres3(p2)
        p3 = self.pool3(x3)
        r3 = self.respath3(x3)
        
        x4 = self.mres4(p3)
        p4 = self.pool4(x4)
        r4 = self.respath4(x4)
        
        x5 = self.mres5(p4)
        
        # Decoder
        u6 = self.up6(x5)
        u6 = torch.cat([u6, r4], dim=1)
        x6 = self.mres6(u6)
        
        u7 = self.up7(x6)
        u7 = torch.cat([u7, r3], dim=1)
        x7 = self.mres7(u7)
        
        u8 = self.up8(x7)
        u8 = torch.cat([u8, r2], dim=1)
        x8 = self.mres8(u8)
        
        u9 = self.up9(x8)
        u9 = torch.cat([u9, r1], dim=1)
        x9 = self.mres9(u9)
        
        out = self.final_conv(x9)
        return out.permute(0, 2, 1)


class MultiResUNetDS(nn.Module):
    """Deeply Supervised MultiResUNet 1D"""
    def __init__(self, length, n_channel=1, base_filters=32):
        super(MultiResUNetDS, self).__init__()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Encoder
        self.mresblock1 = MultiResBlock(base_filters, n_channel)
        self.respath1 = ResPath(base_filters, 4)
        self.mresblock2 = MultiResBlock(base_filters * 2, base_filters)
        self.respath2 = ResPath(base_filters * 2, 3)
        self.mresblock3 = MultiResBlock(base_filters * 4, base_filters * 2)
        self.respath3 = ResPath(base_filters * 4, 2)
        self.mresblock4 = MultiResBlock(base_filters * 8, base_filters * 4)
        self.respath4 = ResPath(base_filters * 8, 1)
        self.mresblock5 = MultiResBlock(base_filters * 16, base_filters * 8)

        # Deep Supervision Layers
        self.level4 = nn.Conv1d(base_filters * 16, 1, kernel_size=1)
        self.level3 = nn.Conv1d(base_filters * 8, 1, kernel_size=1)
        self.level2 = nn.Conv1d(base_filters * 4, 1, kernel_size=1)
        self.level1 = nn.Conv1d(base_filters * 2, 1, kernel_size=1)

        # Decoder
        self.up6 = MultiResBlock(base_filters * 8, base_filters * 16)
        self.up7 = MultiResBlock(base_filters * 4, base_filters * 8)
        self.up8 = MultiResBlock(base_filters * 2, base_filters * 4)
        self.up9 = MultiResBlock(base_filters, base_filters * 2)

        # Final Conv Layer
        self.final_conv = nn.Conv1d(base_filters, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.mresblock1(x)
        p1 = self.pool(c1)
        c1_res = self.respath1(c1)

        c2 = self.mresblock2(p1)
        p2 = self.pool(c2)
        c2_res = self.respath2(c2)

        c3 = self.mresblock3(p2)
        p3 = self.pool(c3)
        c3_res = self.respath3(c3)

        c4 = self.mresblock4(p3)
        p4 = self.pool(c4)
        c4_res = self.respath4(c4)

        c5 = self.mresblock5(p4)
        level4 = self.level4(c5)

        # Decoder
        up6 = self.upsample(c5)
        up6 = torch.cat([up6, c4_res], dim=1)
        up6 = self.up6(up6)
        level3 = self.level3(up6)

        up7 = self.upsample(up6)
        up7 = torch.cat([up7, c3_res], dim=1)
        up7 = self.up7(up7)
        level2 = self.level2(up7)

        up8 = self.upsample(up7)
        up8 = torch.cat([up8, c2_res], dim=1)
        up8 = self.up8(up8)
        level1 = self.level1(up8)

        up9 = self.upsample(up8)
        up9 = torch.cat([up9, c1_res], dim=1)
        up9 = self.up9(up9)

        out = self.final_conv(up9)

        return out, level1, level2, level3, level4