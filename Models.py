import tensorflow as tf
import numpy as np
from keras.layers import Input, DepthwiseConv1D, Lambda, TimeDistributed, GRU, Conv2D, Permute, Reshape, AveragePooling1D, MaxPooling2D,Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Bidirectional, Concatenate
import keras.backend as K

class LogMelLayer(tf.keras.layers.Layer):
    def __init__(self, fft_length, window_length, hop_length, nmels, sr, **kwargs):
        super(LogMelLayer, self).__init__(**kwargs)
        self.fft_length = fft_length
        self.window_length = window_length
        self.hop_length = hop_length
        self.nmels = nmels
        self.sr = sr
        
        
    def call(self, inputs):
        x = tf.squeeze(inputs, axis=-1)
        stft = tf.abs(tf.signal.stft(x, frame_length=self.window_length, frame_step=self.hop_length, fft_length=self.fft_length, pad_end=True))
        filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.nmels,
            num_spectrogram_bins=self.fft_length // 2 + 1,
            sample_rate=self.sr,
            lower_edge_hertz=80,
            upper_edge_hertz=self.sr//2)
        mel = tf.tensordot(stft, filterbank, 1)
        log = 10 * tf.math.log(tf.maximum(mel ** 2, 1e-10)) / tf.math.log(10.0)
        return tf.expand_dims(tf.transpose(log, perm=[0, 1, 2]), axis=-1)  # (batch, mel, time, 1)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.nmels, input_shape[1] // self.hop_length, 1)
    
    def get_config(self):
        config = {
            'fft_length': self.fft_length,
            'window_length': self.window_length,
            'hop_length': self.hop_length,
            'nmels': self.nmels,
            'sr': self.sr
        }
        base_config = super(LogMelLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def LOG2D(n_classes, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb, 
        batchsize_, win_length, filters, kernel_size_1, dropout_rate=0.5, F_size=256, sr=24000):
    # 10, 128, [5, 2, 2], [64, 64], [32], 10, 4096, sr=sr)
    
    x = Input(shape=(win_length, 1), name='input')
    
    # Log Mel 
    
    spec_x = LogMelLayer(fft_length=2048, window_length=1026, hop_length=F_size, nmels=128, sr=24000)(x)
    
    # Conv 2D Blocks
    
    #spec_x = L
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same', name='Cnv2D'+str(_i))(spec_x)
        spec_x = BatchNormalization(axis=1, name='BatchN'+str(_i))(spec_x)
        spec_x = Activation('relu', name='Relu'+str(_i))(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]), name='MaxPool'+str(_i))(spec_x)
        spec_x = Dropout(dropout_rate, name='Dropout'+str(_i))(spec_x)
    
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((win_length//F_size, -1))(spec_x)
    
    num = 1

    for _r in _rnn_nb:
        
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul', name='BiLSTM'+str(num))(spec_x)
        num += 1

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(n_classes))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = tf.keras.Model(inputs=x, outputs=out, name='CRNN_RA')
    _model.compile(tf.keras.optimizers.Adam(learning_rate=0.001)
            , loss='binary_crossentropy'
            , metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(thresholds=0.5)])

    return _model

def RAW2D(n_classes, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb, 
        batchsize_, win_length, filters, kernel_size_1, dropout_rate=0.5, F_size=256, sr=24000):
    # 10, 128, [5, 2, 2], [64, 64], [32], 10, 4096, sr=sr)
    
    x = Input(shape=(win_length, 1), name='input')

    
    
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same', use_bias=False, trainable = True,
                    input_shape=(win_length, 1), name='conv1')
    
    
    
    activation_abs = Activation(K.abs)
    depth_conv = DepthwiseConv1D(kernel_size_1*2, strides=1, padding='same', use_bias=False,
                                depth_multiplier=1, trainable = True, name='conv2')
    
    batch_norm_1 = BatchNormalization()
    batch_norm_2 = BatchNormalization()
    
    activation_sp = Activation('relu')
    activation_re = Activation('relu')
   
    avg_pooling_freq = AveragePooling1D(pool_size=2, data_format='channels_first')
    max_pooling = MaxPooling1D(pool_size=F_size, data_format='channels_last')
    #max_pooling2 = MaxPooling1D(pool_size=pool_size, data_format='channels_last')
    concatenate = Concatenate(axis=-1)
    concatenate2 = Concatenate(axis=-2)
    
    # Learnable filters
    
    X = conv(x)
    X = batch_norm_1(X)
    B = activation_abs(X)
    M = depth_conv(B)
    M = batch_norm_2(M)
    M = activation_sp(M)
    
    # Concatenate Skip connection

    L = concatenate([B, M])
    L = avg_pooling_freq(L)
    L = max_pooling(L)
    L = L[...,tf.newaxis]
    
    
    spec_x = L
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same', name='Cnv2D'+str(_i))(spec_x)
        spec_x = BatchNormalization(axis=1, name='BatchN'+str(_i))(spec_x)
        spec_x = Activation('relu', name='Relu'+str(_i))(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]), name='MaxPool'+str(_i))(spec_x)
        spec_x = Dropout(dropout_rate, name='Dropout'+str(_i))(spec_x)
    
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((win_length//F_size, -1))(spec_x)
    
    num = 1

    for _r in _rnn_nb:
        
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul', name='BiLSTM'+str(num))(spec_x)
        num += 1

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(n_classes))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = tf.keras.Model(inputs=x, outputs=out, name='CRNN_RA')
    _model.compile(tf.keras.optimizers.Adam(learning_rate=0.001)
            , loss='binary_crossentropy'
            , metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(thresholds=0.5)])

    return _model


def PURE1D(n_classes, _rnn_nb, _fc_nb, 
        win_length, dropout_rate=0.25, sr=24000):
    # 10, 128, [5, 2, 2], [64, 64], [32], 10, 4096, sr=sr)
    
    x = Input(shape=(win_length, 1), name='input')

    # First Block
    
    conv = Conv1D(128, 64, strides=2, padding='same', use_bias=True, trainable = True,
                    input_shape=(win_length, 1), name='conv1')(x)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    
    conv_pool = MaxPooling1D(pool_size=8, strides = 4)(conv)
    
    # Second Block 

    conv = Conv1D(128, 32, strides=2, padding='same', use_bias=True, trainable = True,
                    name='conv2')(conv_pool)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    
    conv_pool = MaxPooling1D(pool_size=8, strides = 2)(conv)

    # Third Block 

    conv = Conv1D(128, 16, strides=2, padding='same', use_bias=True, trainable = True,
                    name='conv3')(conv_pool)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    
    conv_pool = MaxPooling1D(pool_size=2, strides = 1)(conv)

    # Fourth Block

    conv = Conv1D(128, 8, strides=2, padding='same', use_bias=True, trainable = True,
                    name='conv4')(conv_pool)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    
    # Fifth Block

    conv = Conv1D(256, 4, strides=2, padding='same', use_bias=True, trainable = True,
                    name='conv5')(conv)
    conv = Activation('relu')(conv)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    
    num = 1
    spec_x = conv
    for _r in _rnn_nb:
        
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul', name='BiLSTM'+str(num))(spec_x)
        num += 1

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(n_classes))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = tf.keras.Model(inputs=x, outputs=out, name='CRNN_RA')
    _model.compile(tf.keras.optimizers.Adam(learning_rate=0.001)
            , loss='binary_crossentropy'
            , metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(thresholds=0.5)])

    return _model

def RAW2D_3_channels(n_classes, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb, 
        batchsize_, win_length, filters, kernel_size_1, dropout_rate=0.5, F_size=256, sr=24000):
    # 10, 128, [5, 2, 2], [64, 64], [32], 10, 4096, sr=sr)
    
    x = Input(shape=(win_length, 1), name='input')

    
    
    conv = Conv1D(filters, kernel_size_1, strides=1, padding='same', use_bias=False, trainable = True,
                    input_shape=(win_length, 1), name='conv1')
    
    # Learnable filters
    
    X = conv(x)
    X = BatchNormalization()(X)
    B = Activation(K.abs)(X)

    # Filter 1
    filter1 = DepthwiseConv1D(kernel_size_1//2, strides=1, padding='same', use_bias=False,
                                depth_multiplier=1, trainable = True, name='conv2_filter1')(B)
    filter1 = BatchNormalization()(filter1)
    filter1 = Activation('relu')(filter1)
    filter1 = MaxPooling1D(pool_size=F_size, data_format='channels_last')(filter1)
    filter1 = filter1[...,tf.newaxis]

    # Filter 2
    filter2 = DepthwiseConv1D(kernel_size_1, strides=1, padding='same', use_bias=False,
                                depth_multiplier=1, trainable = True, name='conv2_filter2')(B)
    filter2 = BatchNormalization()(filter2)
    filter2 = Activation('relu')(filter2)
    filter2 = MaxPooling1D(pool_size=F_size, data_format='channels_last')(filter2)
    filter2 = filter2[...,tf.newaxis]
    
    # Filter 2
    filter3 = DepthwiseConv1D(kernel_size_1*2, strides=1, padding='same', use_bias=False,
                                depth_multiplier=1, trainable = True, name='conv2_filter3')(B)
    filter3 = BatchNormalization()(filter3)
    filter3 = Activation('relu')(filter3)
    filter3 = MaxPooling1D(pool_size=F_size, data_format='channels_last')(filter3)
    filter3 = filter3[...,tf.newaxis]

    # Concatenate
    concat_filter_output = tf.keras.layers.concatenate([filter1, filter2, filter3], axis=-1)
    
    spec_x = concat_filter_output
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv2D(filters=_cnn_nb_filt, kernel_size=(3, 3), padding='same', name='Cnv2D'+str(_i))(spec_x)
        spec_x = BatchNormalization(axis=1, name='BatchN'+str(_i))(spec_x)
        spec_x = Activation('relu', name='Relu'+str(_i))(spec_x)
        spec_x = MaxPooling2D(pool_size=(1, _cnn_pool_size[_i]), name='MaxPool'+str(_i))(spec_x)
        spec_x = Dropout(dropout_rate, name='Dropout'+str(_i))(spec_x)
    
    spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((win_length//F_size, -1))(spec_x)
    
    num = 1

    for _r in _rnn_nb:
        
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul', name='BiLSTM'+str(num))(spec_x)
        num += 1

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f))(spec_x)
        spec_x = Dropout(dropout_rate)(spec_x)

    spec_x = TimeDistributed(Dense(n_classes))(spec_x)
    out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = tf.keras.Model(inputs=x, outputs=out, name='CRNN_RA')
    _model.compile(tf.keras.optimizers.Adam(learning_rate=0.001)
            , loss='binary_crossentropy'
            , metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(thresholds=0.5)])

    return _model

def RAW1D(n_classes, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb, 
        batchsize_, win_length, dropout_rate=0.5, F_size=64, sr=24000):
    # 10, 128, [5, 2, 2], [64, 64], [32], 10, 4096, sr=sr)
    concatenate2 = Concatenate(axis=-2)
    x = Input(shape=(win_length, 1), name='input')

    conv = Conv1D(128, 64, strides=1, padding='same', use_bias=False, trainable = True,
                    input_shape=(win_length, 1))
        
    activation_abs = Activation(K.abs)
    depth_conv = DepthwiseConv1D(128, strides=1, padding='same', use_bias=False,
                                depth_multiplier=1, trainable = True)
    batch_norm_1 = BatchNormalization()
    batch_norm_2 = BatchNormalization()
    
    dropout1 = Dropout(0.5)
    
    activation_sp = Activation('relu')
    activation_re = Activation('relu')
    activation_re2 = Activation('relu')

    
    avg_pooling_freq = AveragePooling1D(pool_size=2, data_format='channels_first')
    max_pooling = MaxPooling1D(pool_size=F_size, data_format='channels_last')
    
    #max_pooling2 = MaxPooling1D(pool_size=pool_size, data_format='channels_last')
    concatenate = Concatenate(axis=-1)
    concatenate2 = Concatenate(axis=-2)
    
    # Learnable filters
    
    # Learnable filters
    
    X = conv(x)
    X = batch_norm_1(X)
    B = activation_abs(X)
    M = depth_conv(B)
    M = batch_norm_2(M)
    M = activation_sp(M)
    
    # Concatenate Skip connection

    L = concatenate([B, M])
    L = avg_pooling_freq(L)
    L = max_pooling(L)
    
    
    spec_x = L
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv1D(filters=512, kernel_size=3, padding='same', name='Cnv1D'+str(_i))(spec_x)
        spec_x = BatchNormalization(name='BatchN'+str(_i))(spec_x)
        spec_x = Activation('relu', name='Relu'+str(_i))(spec_x)
        spec_x = MaxPooling1D(pool_size=4, data_format='channels_first', name='MaxPool'+str(_i))(spec_x)
        spec_x = Dropout(dropout_rate, name='Dropout'+str(_i))(spec_x)
    
    
    num = 1

    
    spec_x = Reshape((win_length//F_size, -1))(spec_x)


    for _r in _rnn_nb:
        
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul', name='BiLSTM'+str(num))(spec_x)
        num += 1

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f), name='Dense_1')(spec_x)

    out = TimeDistributed(Dense(n_classes, activation='sigmoid'), name='Dense_out')(spec_x)

    _model = tf.keras.Model(inputs=x, outputs=out, name='CRNN')
    _model.compile(tf.keras.optimizers.Adam(learning_rate=0.001)
            , loss='binary_crossentropy'
            , metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(thresholds=0.5)])

    return _model

def LOG1D(n_classes, _cnn_nb_filt, _cnn_pool_size, _rnn_nb, _fc_nb, 
        batchsize_, win_length, dropout_rate=0.5, F_size=64, sr=24000):
    # 10, 128, [5, 2, 2], [64, 64], [32], 10, 4096, sr=sr)
    
    x = Input(shape=(win_length, 1), name='input')

    L = LogMelLayer(fft_length=2048, window_length=1026, hop_length=F_size, nmels=128, sr=24000)(x)
    L = tf.squeeze(L, axis=-1)
    spec_x = L
    for _i, _cnt in enumerate(_cnn_pool_size):
        spec_x = Conv1D(filters=512, kernel_size=3, padding='same', name='Cnv1D'+str(_i))(spec_x)
        spec_x = BatchNormalization(name='BatchN'+str(_i))(spec_x)
        spec_x = Activation('relu', name='Relu'+str(_i))(spec_x)
        spec_x = MaxPooling1D(pool_size=4, data_format='channels_first', name='MaxPool'+str(_i))(spec_x)
        spec_x = Dropout(dropout_rate, name='Dropout'+str(_i))(spec_x)
    
    
    num = 1

    #spec_x = Permute((2, 1, 3))(spec_x)
    spec_x = Reshape((win_length//F_size, -1))(spec_x)


    for _r in _rnn_nb:
        
        spec_x = Bidirectional(
            GRU(_r, activation='tanh', dropout=dropout_rate, recurrent_dropout=dropout_rate, return_sequences=True),
            merge_mode='mul', name='BiLSTM'+str(num))(spec_x)
        num += 1

    for _f in _fc_nb:
        spec_x = TimeDistributed(Dense(_f), name='Dense_1')(spec_x)
        #spec_x = TimeDistributed(Dropout(dropout_rate))(spec_x)

    out = TimeDistributed(Dense(n_classes, activation='sigmoid'), name='Dense_out')(spec_x)
    #out = Activation('sigmoid', name='strong_out')(spec_x)

    _model = tf.keras.Model(inputs=x, outputs=out, name='CRNN')
    _model.compile(tf.keras.optimizers.Adam(learning_rate=0.001)
            , loss='binary_crossentropy'
            , metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision(thresholds=0.5)])

    return _model
