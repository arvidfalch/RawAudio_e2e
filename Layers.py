import keras
import keras.backend as K
from Utils import slicing
import tensorflow as tf
from keras import layers 
from keras.layers import Layer 
from keras.layers import InputSpec

from keras.utils import conv_utils
from keras import activations, initializers, regularizers, constraints

import numpy as np

# Log mel layer
# Code from https://gist.github.com/keunwoochoi/f4854acb68acf791a49a051893bcd23b

class LogMelgramLayer(tf.keras.layers.Layer):
    def __init__(
        self, num_fft, hop_length, num_mels, sample_rate, f_min, f_max, eps, **kwargs
    ):
        super(LogMelgramLayer, self).__init__(**kwargs)
        self.num_fft = num_fft
        self.hop_length = hop_length
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max
        self.eps = eps
        self.num_freqs = num_fft // 2 + 1
        lin_to_mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mels,
            num_spectrogram_bins=self.num_freqs,
            sample_rate=self.sample_rate,
            lower_edge_hertz=self.f_min,
            upper_edge_hertz=self.f_max,
        )

        self.lin_to_mel_matrix = lin_to_mel_matrix

    def build(self, input_shape):
        self.non_trainable_weights.append(self.lin_to_mel_matrix)
        super(LogMelgramLayer, self).build(input_shape)

    def call(self, input):
        """
        Args:
            input (tensor): Batch of mono waveform, shape: (None, N)
        Returns:
            log_melgrams (tensor): Batch of log mel-spectrograms, shape: (None, num_frame, mel_bins, channel=1)
        """

        def _tf_log10(x):
            numerator = tf.math.log(x)
            denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
            return numerator / denominator

        stfts = tf.signal.stft(
            input,
            frame_length=self.num_fft,
            frame_step=self.hop_length,
            pad_end=False,  # librosa test compatibility
        )
        mag_stfts = tf.abs(stfts)

        melgrams = tf.tensordot(  # assuming channel_first, so (b, c, f, t)
            tf.square(mag_stfts), self.lin_to_mel_matrix, axes=[2, 0]
        )
        log_melgrams = _tf_log10(melgrams + self.eps)
        return tf.expand_dims(log_melgrams, 3)

    def get_config(self):
        config = {
            'num_fft': self.num_fft,
            'hop_length': self.hop_length,
            'num_mels': self.num_mels,
            'sample_rate': self.sample_rate,
            'f_min': self.f_min,
            'f_max': self.f_max,
            'eps': self.eps,
        }
        base_config = super(LogMelgramLayer, self).get_config()
        return dict(list(config.items()) + list(base_config.items()))



def normalize_data_format(value):
    if value is None:
        value = K.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format


class Conv1D_tied(Layer):
        
    def __init__(self, filters,
                 kernel_size,
                 tied_to,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 input_length=None,
                 data_format='channels_last',
                 **kwargs):

        if padding not in {'valid', 'same', 'causal'}:
            raise Exception('Invalid padding mode for Convolution1D:', padding)
        
        super(Conv1D_tied, self).__init__(**kwargs)
        
        self.rank = 1
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = normalize_data_format('channels_last')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.tied_to = tied_to
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)

        
    def build(self, input_shape):
        
            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1
            if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
           
            if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
                
                
            input_dim = input_shape[channel_axis]
            kernel_shape = self.kernel_size + (input_dim, self.filters)

            if self.use_bias:
                self.bias = self.add_weight(shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
                self.trainable_weights = [self.bias]
            else:
                self.bias = None
            self.input_spec = InputSpec(ndim=self.rank + 2,
                                        axes={channel_axis: input_dim})
            self.built = True        

            
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
        return (input_shape[0], self.filters) + tuple(new_space)
    
    
            
    def call(self, x, mask=None):
        
        x = K.expand_dims(x, -1)  # add a dimension of the right
        x = K.permute_dimensions(x, (0, 3, 1, 2))

        W = self.tied_to.kernel   
        W = K.expand_dims(W, -1)
        W = tf.transpose(W, (1, 0, 2, 3))
        
        output = K.conv2d(x, W, 
                          strides=(self.strides,self.strides),
                          padding=self.padding,
                          data_format=self.data_format)
        if self.bias:
            output += K.reshape(self.bias, (1, self.filters, 1, 1))
        output = K.squeeze(output, 3)  # remove the dummy 3rd dimension
        output = K.permute_dimensions(output, (0, 2, 1))
        output = self.activation(output)
        return output


    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'tied_to': self.tied_to.name
        }
        base_config = super(Conv1D_tied, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


    
    
class Conv1D_local(Layer):

    # Locally-connected 1D convolutional layer. Performs one-to-one convolutions to input feature map.
        
    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 data_format='channels_last',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 input_dim=None,
                 input_length=None,
                 **kwargs):

        if padding not in {'valid', 'same', 'causal'}:
            raise Exception('Invalid padding mode for Convolution1D:', padding)
        
        super(Conv1D_local, self).__init__(**kwargs)
        
        self.rank = 1
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, self.rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, self.rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = normalize_data_format('channels_last')
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, self.rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.input_dim = input_dim
        self.input_length = input_length
        if self.input_dim:
            kwargs['input_shape'] = (self.input_length, self.input_dim)
        
        

        
    def build(self, input_shape):
        

            if self.data_format == 'channels_first':
                channel_axis = 1
            else:
                channel_axis = -1
            if input_shape[channel_axis] is None:
                raise ValueError('The channel dimension of the inputs '
                                 'should be defined. Found `None`.')
                
                
            input_dim = input_shape[channel_axis]
            kernel_shape = self.kernel_size + (1, self.filters)
            
            self.kernel = self.add_weight(shape=kernel_shape,
                                          initializer=self.kernel_initializer,
                                          name='kernel',
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint)
            if self.use_bias:
                self.bias = self.add_weight(shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            name='bias',
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
            else:
                self.bias = None
            # Set input spec.
            self.input_spec = InputSpec(ndim=self.rank + 2,
                                        axes={channel_axis: input_dim})
            self.built = True        
       

    def compute_output_shape(self, input_shape):
        
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
                
        return (input_shape[0], input_shape[1], self.filters) 
    
    def call(self, inputs):
        

        x = tf.split(inputs, self.filters, axis = 2)
        W = tf.split(self.kernel, self.filters, axis = 2)
        outputs = []
        
        for i in range(self.filters):
            output = K.conv1d(x[i], W[i],
                              strides=self.strides[0],
                              padding=self.padding,
                              data_format=self.data_format,
                              dilation_rate=self.dilation_rate[0])
    
    
            outputs.append(output)
    
        outputs = K.concatenate(outputs,axis=-1)
        
        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        
        return outputs


    
    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Conv1D_local, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
    
    
class SAAF(Layer):
    '''[references]
    [1] ConvNets with Smooth Adaptive Activation Functions for Regression, Hou, Le
    and Samaras, Dimitris and Kurc, Tahsin M and Gao, Yi and Saltz, Joel H,
    Artificial Intelligence and Statistics, 2017
     '''
    def __init__(self,
                 break_points,
                 break_range = 0.2,
                 magnitude = 1.0,
                 order = 2,
                 tied_feamap = True,
                 kernel_initializer = 'random_normal',
                 kernel_regularizer = None,
                 kernel_constraint = None,
                 **kwargs):
        super(SAAF, self).__init__(**kwargs)
        self.break_range = break_range
        self.break_points = list(np.linspace(-self.break_range, self.break_range, break_points, dtype=np.float32))
        self.num_segs = int(len(self.break_points) / 2)
        self.magnitude = float(magnitude)
        self.order = order
        self.tied_feamap = tied_feamap
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        
        

    def build(self, input_shape):

        if self.tied_feamap:
            kernel_dim = (self.num_segs + 1, input_shape[2])
        else:
            kernel_dim = (self.num_segs + 1,) + input_shape[2::]
            
        self.kernel = self.add_weight(shape=kernel_dim,
                                     name='kernel',
                                     initializer=self.kernel_initializer,
                                     regularizer=self.kernel_regularizer,
                                     constraint=self.kernel_constraint)
        
    def basisf(self, x, s, e):
        cpstart = tf.cast(tf.less_equal(s, x), tf.float32)
        cpend = tf.cast(tf.greater(e, x), tf.float32)
        if self.order == 1:
            output = self.magnitude * (0 * (1 - cpstart) + (x - s) * cpstart * cpend + (e - s) * (1 - cpend))
        else:
            output = self.magnitude * (0 * (1 - cpstart) + 0.5 * (x - s)**2 * cpstart
                                     * cpend + ((e - s) * (x - e) + 0.5 * (e - s)**2) * (1 - cpend))
        
        return tf.cast(output, tf.float32)
        
        self.built = True

    def call(self, x):
         
        output = tf.zeros_like(x)
        
        if self.tied_feamap:
            output += tf.multiply(x,self.kernel[-1])
        else:
            output += tf.multiply(x,self.kernel[-1])
        for seg in range(0, self.num_segs):
            if self.tied_feamap:
                output += tf.multiply(self.basisf(x, self.break_points[seg * 2], self.break_points[seg * 2 + 1]), 
                                      self.kernel[seg])
            else:
                output += tf.multiply(self.basisf(x, self.break_points[seg * 2], self.break_points[seg * 2 + 1]), 
                                      self.kernel[seg])

        return output

    def get_config(self):
        config = {
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'break_points': self.break_points,
            'magnitude': self.magnitude,
            'order': self.order,
            'tied_feamap': self.tied_feamap

        }
        base_config = super(SAAF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))   
    
    
class Dense_local(Layer):
    # Locally-connected dense layer. Applies a different fully connected layer to each channel of the input feature. 
    
    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(Dense_local, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
         

    def build(self, input_shape):
        
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.split = input_shape[1]
        kernels = []
       
        for i in range(input_shape[-2]):
            
            kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
            kernels.append(kernel)
        
        self.kernel = K.concatenate(kernels,axis=-1)
   
        if self.use_bias:
            biases = []
            for i in range(input_shape[-2]):
                bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
                
                biases.append(bias)
                
            self.bias = K.concatenate(biases,axis=-1)
        else:
            self.bias = None  
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):

        #unstacked = K.tf.unstack(inputs, axis=1)  
        unstacked = K.tf.split(inputs, self.split, axis = 1)
        W = tf.split(self.kernel, len(unstacked), axis = 1)
        b = tf.split(self.bias, len(unstacked), axis = 0)
        #kernels = K.tf.unstack(self.kernels, axis=1)
        #biases = K.tf.unstack(self.biases, axis=0)
        outputs = []
        for i,j in enumerate(unstacked):
            
            output = K.dot(j, W[i])
        
            if self.use_bias:
                output = K.bias_add(output, b[i])
                
            if self.activation is not None:
                output = self.activation(output)
            
            outputs.append(output)
            
        outputs = K.concatenate(outputs,axis=1)
                
        #outputs = K.stack(outputs, axis=1)
        return outputs
        #return tf.zeros(self.units)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(Dense_local, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    

    
    
    
class Slice(Layer):

    def __init__(self, selector, output_shape, **kwargs):
        self.selector = selector
        self.desired_output_shape = output_shape
        super(Slice, self).__init__(**kwargs)

    def call(self, x, mask=None):

        selector = self.selector
        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            x = keras.backend.permute_dimensions(x, [0, 2, 1])
            selector = (self.selector[1], self.selector[0])

        y = x[selector]

        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            y = keras.backend.permute_dimensions(y, [0, 2, 1])

        return y


    def compute_output_shape(self, input_shape):

        output_shape = (None,)
        for i, dim_length in enumerate(self.desired_output_shape):
            if dim_length == Ellipsis:
                output_shape = output_shape + (input_shape[i+1],)
            else:
                output_shape = output_shape + (dim_length,)
                
        return output_shape    
    
    
#Generators for training. dry and wet tensors should be of tensor shape (number_of_recordings, number_of_samples, 1) 

# Generator for pretraining and CAFx.

'''class Generator(Sequence):

    def __init__(self, x_set, y_set, win_length, hop_length, win = False):
        self.x, self.y = x_set, y_set
        self.win_length = win_length
        self.hop_length = hop_length
        self.batch_size = int(self.x.shape[1] / self.hop_length) + 1
        self.win = win

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        batch_x = np.zeros((self.batch_size, self.win_length, 1))
        
        batch_y = np.zeros((self.batch_size, self.win_length, 1))
        
        
        x_w = self.x[idx].reshape(len(self.x[idx]))
        y_w = self.y[idx].reshape(len(self.y[idx]))

        
        x_w = slicing(x_w, self.win_length, self.hop_length, windowing = self.win)
        y_w = slicing(y_w, self.win_length, self.hop_length, windowing = self.win)
        
        for i in range(self.batch_size):
            
            batch_x[i] = x_w[i].reshape(self.win_length,1)  
            batch_y[i] = y_w[i].reshape(self.win_length,1) 

            
        return batch_x, batch_y
    

# Generator for WaveNet.

class GeneratorWaveNet(Sequence):

    def __init__(self, x_set, y_set, in_length, out_length, hop_length, win = False):
        self.x, self.y = x_set, y_set
        self.in_length = in_length
        self.out_length = out_length
        self.hop_length = hop_length
        self.batch_size = int(self.x.shape[1] / self.hop_length) + 1
        self.win = win
        self.trim = (in_length - out_length)//2

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        batch_x = np.zeros((self.batch_size, self.in_length, 1))
        
        batch_y = np.zeros((self.batch_size, self.out_length, 1))
        
        
        x_w = self.x[idx].reshape(len(self.x[idx]))
        y_w = self.y[idx].reshape(len(self.y[idx]))

        
        x_w = slicing(x_w, self.in_length, self.hop_length, windowing = self.win)
        y_w = slicing(y_w, self.in_length, self.hop_length, windowing = self.win)
        
        for i in range(self.batch_size):
            
            batch_x[i] = x_w[i].reshape(self.in_length,1)  
            batch_y[i] = y_w[i].reshape(self.in_length,1)[self.trim:self.trim+self.out_length] 

            
        return batch_x, batch_y
    

# Generator for CRAFx and CWAFx. Audio samples should be already zero padded at the end (0.5seconds.)    
    
class GeneratorContext(Sequence):

    def __init__(self, x_set, y_set, context, win_length, hop_length, win = False, win_input = None):
        self.x, self.y = x_set, y_set
        self.win_length = win_length
        self.hop_length = hop_length
        self.batch_size = int(self.x.shape[1] / self.hop_length) + 1
        self.win_output = win
        if win_input == None:
            self.win_input = win
        else:
            self.win_input = win_input
        self.context = context

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        
        batch_x = []
        for i in range(self.context*2+1):
            batch_x.append(np.zeros((self.batch_size, self.win_length, 1)))
        batch_y = np.zeros((self.batch_size, self.win_length, 1))
        
        
        x_w = self.x[idx].reshape(len(self.x[idx]))
        y_w = self.y[idx].reshape(len(self.y[idx]))

        
        x_w = slicing(x_w, self.win_length, self.hop_length, windowing = self.win_input)

        x_w = np.pad(x_w, ((self.context, self.context),(0, 0)), 'constant', constant_values=(0))
        a = []
        for i in range(x_w.shape[0]):
            a.append(x_w[i:i+self.context*2+1])
        del a[-self.context*2:]
        a = np.asarray(a)
       
        y_w = slicing(y_w, self.win_length, self.hop_length, windowing = self.win_output)
        
        for i in range(self.batch_size):
            
            for j in range(self.context*2+1):
                batch_x[j][i] = a[:,j,:][i].reshape(self.win_length,1)
                       
            batch_y[i] = y_w[i].reshape(self.win_length,1) 
            
        batch_x = np.swapaxes(np.asarray(batch_x), 0, 1)
        
        return batch_x, batch_y  '''