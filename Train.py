import numpy as np
import os
import tensorflow as tf
from keras.layers import Input, DepthwiseConv1D, Lambda, TimeDistributed, GRU, Conv2D, Permute, Reshape, AveragePooling1D, MaxPooling2D,Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, LSTM, Bidirectional, Concatenate
import keras.backend as K
import argparse
import Models

parser = argparse.ArgumentParser(description='Get the parameters for training')
parser.add_argument('--batchsize', metavar = 'batchsize', type=int, default=10, help='batchsize')
parser.add_argument('--epochs', metavar = 'epochs', type=int, default=1, help='epochs')
parser.add_argument('--model_name', metavar = 'model_name', type=str, default='model', help='model_name')
parser.add_argument('--load_model', metavar = 'load_model', type=int, default=0, help='load_model')
parser.add_argument('--frame_size', metavar='frame_size', type=int, default=256, help='analysis frame size in samples')
parser.add_argument('--model_type', metavar='model_type', type=int, default=0,
                     help= '0 = RAW2D, 1 = LOG2D, 2 = RAW1D, 3 = LOG1D, 4 = PURE1D (only on analysis frame size 256)')
args = parser.parse_args()

epochs_ = args.epochs # Number of epochs (each epoch is a full pass through the full dataset)
batchsize = args.batchsize # Batchsize for training
model_name = args.model_name # Model name, make sure it exists if load_model = 1, or unique if load_model = 0
load_model = args.load_model # 0 = False, 1 = True
analysis_frame_size = args.frame_size # 64 or 256 in Thesis
model_type = args.model_type # Which model to train: 0 = RAW2D, 1 = LOG2D, 2 = RAW1D, 3 = LOG1D, 4 = PURE1D (only on analysis frame size 256)

sr = 24000

tf.config.list_physical_devices('GPU')

tf.keras.backend.clear_session()

path = './Models/' 

try:
    os.mkdir('./Models/' + model_name + '/', mode = 0o777)
except OSError:
    pass

save_dir = path + model_name 

if model_type == 0:
    _model = Models.RAW2D(10, 128, [5, 2, 2], [64, 64], [32], 10, 40960, 128, 64, dropout_rate=0.5, sr=sr, F_size=analysis_frame_size)
    _model.summary()
elif model_type == 1:
    _model = Models.LOG2D(10, 128, [5, 2, 2], [64, 64], [32], 10, 40960, 128, 64, dropout_rate=0.5, sr=sr, F_size=analysis_frame_size)
    _model.summary()
elif model_type == 2:
    _model = Models.RAW1D(10, 128, [5, 2, 2], [64, 64], [32], 10, 40960, dropout_rate=0.5, sr=sr, F_size=analysis_frame_size)
    _model.summary()
elif model_type == 3:
    _model = Models.LOG1D(10, 128, [5, 2, 2], [64, 64], [32], 10, 40960, dropout_rate=0.5, sr=sr, F_size=analysis_frame_size)
    _model.summary()
elif model_type == 4:
    _model = Models.PURE1D(10, [64, 64], [32], 40960, dropout_rate=0.5, sr=sr)
    _model.summary()
else:
    _model = Models.RAW2D(10, 128, [5, 2, 2], [64, 64], [32], 10, 40960, 128, 64, dropout_rate=0.5, sr=sr, F_size=analysis_frame_size)
    _model.summary()

print('Batch size: ', batchsize)
print('Epochs: ', epochs_)

if load_model == 0:
    print('Training from scratch')
elif load_model == 1:
    print('Loading model')
    _model = tf.keras.models.load_model(save_dir)
    #_model.load_weights(save_dir + 'checkpoint')


checkpoint_path = save_dir + '/' + 'checkpoint'
callback_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                 verbose=1)

# Linear decreasing learning rate
def lr_(epochs, starting_lr, ending_lr):
    x = np.linspace(starting_lr, ending_lr, epochs)
    return x
    
lr_sched = lr_(epochs_, 0.001, 0.0008)
# Set to true if using LR schedule
lr_schedule = False


for i in range(epochs_):
    print('Epoch: ', i+1)

    for b in range(4):
        tf.keras.backend.clear_session()
        if load_model == 0 and i == 0 and b == 0:
            print('Training from scratch')
        else:
            _model = tf.keras.models.load_model(save_dir)
            print('model loaded')
        if lr_schedule is True: 
            K.set_value(_model.optimizer.learning_rate, lr_sched[i])
        else:
            pass
        print('Learning rate: {}'.format(lr_sched[i]))
        TRAIN = np.load('DESED{}.npz'.format(b+1), allow_pickle= True)
        dmp = TRAIN
        features, labels, label_names = dmp['arr_0'],  dmp['arr_1'],  dmp['arr_2']
        features = np.reshape(features, (-1, features.shape[1]*features.shape[2],1))
        
        if analysis_frame_size == 256:
            reshaped_arr = labels.reshape((-1, ((labels.shape[1]*labels.shape[2])//analysis_frame_size), 4, 10))
            max_values = np.amax(reshaped_arr, axis=2)
            labels = max_values.reshape((labels.shape[0], -1, 10))
            
        else:
            pass
        features = features[0:14990]
        labels = labels[0:14990]
        VAL = np.load('DESED_VAL{}.npz'.format(b+1), allow_pickle= True)
        dmp2 = VAL
        features_val, labels_val, label_names_val = dmp2['arr_0'],  dmp2['arr_1'],  dmp2['arr_2']
        features_val = np.reshape(features_val, (-1, features_val.shape[1]*features_val.shape[2],1))
        if analysis_frame_size == 256:
            reshaped_arr = labels_val.reshape((labels_val.shape[0], -1, 4, 10))
            max_values = np.amax(reshaped_arr, axis=2)
            labels_val = max_values.reshape((labels_val.shape[0], -1, 10))
            
        else:
            pass

        _model.fit(
            features,
            labels,
            validation_data=[features_val, labels_val],
            batch_size=batchsize,
            shuffle=True,
            epochs=1,
            verbose=1,
            callbacks = [callback_stop, cp_callback])
        tf.keras.models.save_model(_model, save_dir)
        del TRAIN, dmp, features, labels, label_names, VAL, dmp2, features_val, labels_val, label_names_val
        del _model

print('finished training')
print('Done')




