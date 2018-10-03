from keras.models import Model
from keras.layers import Input,Conv1D, Dense, MaxPool1D, Activation, AvgPool1D,GlobalAveragePooling1D
from keras.layers import Flatten, Add, Concatenate, Dropout, BatchNormalization
from keras.regularizers import l2

def ResidualBlock(filters,kernel_size,strides,pool_size,inputs, l_2=0.0, activation='relu', kernel_initializer='he_normal'):
    path1 = MaxPool1D(pool_size=pool_size, padding = 'same', strides = strides)(inputs)
    
    path2 = BatchNormalization()(inputs)
    path2 = Activation(activation=activation)(path2)
    path2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = strides, padding = 'same',
                   kernel_regularizer = l2(l_2),
                   kernel_initializer = kernel_initializer)(path2)
    path2 = BatchNormalization()(path2)
    path2 = Activation(activation=activation)(path2)
    path2 = Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same', 
                   kernel_regularizer = l2(l_2),
                   kernel_initializer = kernel_initializer)(path2)
    path2 = Add()([path2, path1])
    return path2

def build_model(length=300, n_channel=2, n_classes=2, filters=64, kernel_size=3, layers = 10,
                activation='relu',kernel_initializer = 'he_normal', l_2=0.0):    
    sig_inp =  Input(shape=(length, n_channel))  
    inp = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding="same", 
                 kernel_regularizer=l2(l_2))(sig_inp)
    inp = BatchNormalization()(inp)
    inp = Activation(activation=activation)(inp)
    inp_max = MaxPool1D(pool_size=2)(inp)

    l1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=2, padding="same",
                kernel_regularizer=l2(l_2))(inp)
    l1 = BatchNormalization()(l1)
    l1 = Activation(activation=activation)(l1)
    l1 = Conv1D(filters=filters, kernel_size=kernel_size, strides=1, padding="same",
                kernel_regularizer=l2(l_2))(l1)

    new_inp = Add()([l1,inp_max])

    for i in range(layers):
    # every alternate residual block subsample its input by a factor of 2
        if i % 2 == 1:
            pool_size = 2
            strides = 2
        else:
            pool_size = 1
            strides = 1
        # incremented filters    
        if i % 4 == 3:
            filters = 64*int(i//4 + 2)
            new_inp = Conv1D(filters = filters, kernel_size = kernel_size, strides = 1, padding = 'same',
                             kernel_regularizer=l2(l_2),
                             kernel_initializer = kernel_initializer)(new_inp)
        new_inp = ResidualBlock(filters,kernel_size,strides,pool_size,new_inp, l_2=l_2)

    new_inp = GlobalAveragePooling1D()(new_inp)
    new_inp = BatchNormalization()(new_inp)
    new_inp = Dense(128, kernel_regularizer=l2(l_2))(new_inp) 
    new_inp = BatchNormalization()(new_inp)
    new_inp = Activation(activation=activation)(new_inp)
    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(l_2))(new_inp)
    
    model = Model(inputs=[sig_inp],outputs=[out])
    return model

