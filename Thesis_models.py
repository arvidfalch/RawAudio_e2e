
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
from keras.layers import Input, Conv1D, Conv2D, Dense, Activation, Concatenate, TimeDistributed, Lambda, Reshape, Dropout, Permute
from keras.layers import Multiply, Add, UpSampling1D, MaxPooling1D, BatchNormalization, Bidirectional, LSTM, GRU, MaxPooling2D
from Layers import Conv1D_local, Dense_local, SAAF, Conv1D_tied, Slice



def Frontend(batchsize_, win_length, filters, kernel_size_1, melspec=False, 
            output_dim=64, CRNN_output=False):
    # CRNN_output adds channel dimension to the output (1 channel, data_format=channel last) 
    # for use in any Conv2D model
    x = Input(shape=(batchsize_, win_length, 1), name='input')

    
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same',
                       kernel_initializer='lecun_uniform', input_shape=(win_length, 1))
    
    activation_abs = Activation(K.abs, name='conv_activation') 
    # Original CAFx model uses softplus activation function
    activation_sp = tf.keras.layers.ReLU()
    max_pooling = MaxPooling1D(pool_size=win_length//output_dim, data_format='channels_last')

    conv_smoothing = Conv1D_local(filters, kernel_size_1*2, strides=1, padding='same',
                                  kernel_initializer='lecun_uniform')
    
    
    X = TimeDistributed(conv, name='conv')(x)
    X_abs = TimeDistributed(activation_abs, name='conv_activation')(X)
    M = TimeDistributed(conv_smoothing, name='conv_smoothing')(X_abs)
    M = TimeDistributed(activation_sp, name='conv_smoothing_activation')(M)
    frontend_output = TimeDistributed(max_pooling, name='max_pooling')(M)
    if CRNN_output is True:
        frontend_output = frontend_output[..., tf.newaxis]
    else:
        pass
    

    model = tf.keras.Model(inputs=[x], outputs=[frontend_output], name='Frontend')

    return model


def LSTM_backend(batchsize_, win_length, filters, kernel_size_1, n_of_classes, 
            melspec=False, output_dim=64, frame_level_classification=False, dense_units=32,
            activation='tanh'):
   
    frontend = Frontend(batchsize_, win_length, filters, kernel_size_1, melspec=melspec, output_dim=output_dim)

    bi_rnn = Bidirectional(LSTM(filters//2, activation=activation, stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1, name='BiLSTM'))
    bi_rnn1 = LSTM(filters//2, activation=activation, stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1, name='LSTM_1')
    if frame_level_classification is True:
        bi_rnn2 = LSTM(filters//2, activation=activation, stateful=False,
                                 return_sequences=False, dropout=0.1,
                                 recurrent_dropout=0.1, name='LSTM_2')
    elif frame_level_classification is False:
        bi_rnn2 = LSTM(filters//2, activation=activation, stateful=False,
                                 return_sequences=True, dropout=0.1,
                                 recurrent_dropout=0.1, name='LSTM_2')
    
    


    Z = TimeDistributed(bi_rnn, name='BiLSTM')(frontend.output)
    Z = TimeDistributed(bi_rnn1, name='LSTM1')(Z)
    Z = TimeDistributed(bi_rnn2, name='LSTM2')(Z)
    
    z = TimeDistributed(keras.layers.Dense(dense_units, activation=activation, name='Dense_Xtra'))(Z)
    y = TimeDistributed(keras.layers.Dense(n_of_classes, name='Dense_layer', activation='sigmoid'))(z)
   

    model = tf.keras.Model(inputs=[frontend.input], outputs=[y], name='LSTM')
    
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True)


    # Compile the model
    model.compile(tf.keras.optimizers.Adam(learning_rate=lr_schedule,)
                    , loss='binary_crossentropy', metrics='accuracy') 


    return model




def CRNN(n_classes, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb, 
        batchsize_, win_length, filters, kernel_size_1, dropout_rate=0.1, output_dim=64, melspec=False):
# Original code from https://github.com/sharathadavanne/sed-crnn/blob/master/sed.py

    frontend = Frontend(batchsize_, win_length, filters, kernel_size_1, melspec=melspec, output_dim=output_dim, CRNN_output=True)
    
    spec_start = frontend.output
    spec_x = spec_start
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = TimeDistributed(Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same'), name='Cnv2D'+str(_i))(spec_x)
        spec_x = TimeDistributed(BatchNormalization(axis=1), name='BatchN'+str(_i))(spec_x)
        spec_x = TimeDistributed(Activation('relu'), name='Relu'+str(_i))(spec_x)
        spec_x = TimeDistributed(MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]), name='MaxPool'+str(_i)))(spec_x)
        spec_x = TimeDistributed(Dropout(dropout_rate), name='Dropout'+str(_i))(spec_x)
    
    spec_x = TimeDistributed(Permute((2, 1, 3)))(spec_x)
    spec_x = TimeDistributed(Reshape((output_dim, -1)))(spec_x)

    num = 1
    for _r in _rnn_nb:
        
        spec_x = TimeDistributed(Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul'), name='BiLSTM'+str(num))(spec_x)
        num += 1

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(n_classes))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = tf.keras.Model(inputs=frontend.input, outputs=out, name='CRNN')
    _model.compile(optimizer='Adam', loss='binary_crossentropy', metrics='accuracy')

    return _model