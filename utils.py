import os
import keras
import random
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences

MEAN_B, STD_B = 138.712, 16.100
MEAN_M, STD_M =  36.346, 25.224

def data_preprocess(x, random_noise=True, normalized=True):
    length = x.shape[0]
    # get x and then remove zeros (no info)
    x = x[(x[:,0] > 0.0) * (x[:,1] > 0.0)]
    
    if normalized:
        x[:,0] = (x[:,0] - MEAN_B)/STD_B
        x[:,1] = (x[:,1] - MEAN_M)/STD_M
    
    # add random_noise
    if random_noise:
        # x1, x2 = np.mean(x, axis=0)
        noise = np.array([[random.gauss(mu=0, sigma=0.01), 
                           random.gauss(mu=0, sigma=0.01)] for _ in range(x.shape[0])], dtype=np.float32)
        x = x + noise

    # transpose to (n_channel, arbitrary length), then padd to (n_channel, length)
    x = pad_sequences(np.transpose(x), padding='post', value=0.0, maxlen=length, dtype=np.float)

    # transpose back to original shape and store
    return np.transpose(x)


def k_slice_X(Xvalid, Yvalid, k_slice=5, length=300, class_weight = {}):
    """
    # moving across a sequence, we slice out "k_slice" segments with a constant interval
    # in order to increase validation data
    # ex:  |------------------|
    # 1    |------|
    # 2       |------|
    # 3          |------|
    # 4             |------|
    # 5                |------|
    """
    if not class_weight:
        class_weight = dict()
        for i in range(Yvalid.shape[1]):
            class_weight[i] = 1

    intvl = (Xvalid.shape[1] - length)//k_slice

    Xtest = np.empty((Xvalid.shape[0]*k_slice, length, Xvalid.shape[2]))
    Ytest = np.empty((Yvalid.shape[0]*k_slice, Yvalid.shape[1]))
    Wtest = np.empty((Yvalid.shape[0]*k_slice,))

    for k in range(k_slice):
        st = k * Xvalid.shape[0]
        for i in range(Xvalid.shape[0]):
            # print(st+i)
            Xtest[st+i,:,:] = data_preprocess(Xvalid[i,k*intvl:(k*intvl+length),:])
            Ytest[st+i,:] = Yvalid[i,:]
            Wtest[st+i] = class_weight[np.argmax(Yvalid[i,:])]
    return Xtest, Ytest, Wtest

def get_n_zeros(d):
    n_zeros = list()
    for i in range(d.shape[0]):
        n_zeros.append(sum(d[i,:] ==0))
    return np.array(n_zeros)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_keras_csv_logger(csv_logger, save_dir='', accuracy=False):
    loss = pd.read_table(csv_logger.filename, delimiter=',')
    print('min val_loss {0} at epoch {1}'.format(min(loss.val_loss), np.argmin(loss.val_loss)))
    plt.plot(loss.epoch, loss.loss, label='loss')
    plt.plot(loss.epoch, loss.val_loss, label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.close()

    if accuracy:
        print('max val_accu {0} at epoch {1}'.format(max(loss.val_acc), np.argmax(loss.val_acc)))
        plt.plot(loss.epoch, loss.acc, label='accu')
        plt.plot(loss.epoch, loss.val_acc, label='val_accu')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.savefig(os.path.join(save_dir, 'accu.png'))
        plt.close()

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, inputs, labels, batch_size=32, length=600, shuffle=True, random_noise=True):
        'Initialization'
        assert length <= inputs.shape[1], 'length should not exceed inputs.shape[1]' 
        
        self.length = length # (signal_length, num_channel)
        self.batch_size = batch_size
        self.inputs = inputs # (num_signal, signal_length, num_channel)
        self.labels = labels
        self.shuffle = shuffle
        self.random_noise = random_noise
        
        self.n_sample = inputs.shape[0]
        self.n_channel = inputs.shape[2]
        self.n_classes = labels.shape[1]
        self.indexes = np.arange(self.n_sample)
        self.on_epoch_end()
        if length < inputs.shape[1]:
            self.random_crop = True
        elif length == inputs.shape[1]:
            self.random_crop = False
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index = index % np.ceil(self.n_sample/self.batch_size)
        if (index+1)*self.batch_size > len(self.indexes):
            ed = len(self.indexes)
            st = ed - self.batch_size
            self.on_epoch_end()
        else:
            ed = (index+1)*self.batch_size
            st = index * self.batch_size
        indexes = self.indexes[int(st):int(ed)]
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.length, self.n_channel), dtype=np.float)
        Y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        if self.random_crop:
            for i, ID in enumerate(indexes):
                st = random.choice(np.arange(0, self.inputs.shape[1] - self.length))
                x = self.inputs[ID, st:(st+self.length),:]
                X[i,] = self.__data_preprocess(x)
                
                Y[i,] = self.labels[ID]
        else:
            for i, ID in enumerate(indexes):
                x = self.inputs[ID, :, :]
                X[i,] = self.__data_preprocess(x)
                
                Y[i,] = self.labels[ID]
        
        return X, Y


    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_sample)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def __data_preprocess(self,x):
        # get x and then remove zeros (no info)
        x = x[(x[:,0] > 0.0) * (x[:,1] > 0.0)]
        
        x[:,0] = (x[:,0] - MEAN_B)/STD_B
        x[:,1] = (x[:,1] - MEAN_M)/STD_M

        if self.random_noise:
            # x1, x2 = np.mean(x, axis=0)
            noise = np.array([[random.gauss(mu=0, sigma=0.01), 
                               random.gauss(mu=0, sigma=0.01)] for _ in range(x.shape[0])], dtype=np.float)
            x = x + noise

        # transpose to (n_channel, arbitrary length), then padd to (n_channel, length)
        x = pad_sequences(np.transpose(x), padding='post', value=0.0, maxlen=self.length, dtype=np.float)
        
        # transpose back to original shape and store
        return np.transpose(x)