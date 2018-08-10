from __future__ import print_function

import glob

import numpy as np

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import imdb
from keras.layers import (Activation, Dense, Dropout, Embedding,
                          GlobalMaxPooling1D)
from keras.layers.convolutional import Conv1D
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import plot_model

np.random.seed(42)  # for reproducibility


# set parameters:
max_features = 5000
maxlen = 400
batch_size = 32
embedding_dims = 50
nb_filter = 250
filter_length = 3
hidden_dims = 250
nb_epoch = 100
test_ratio = 0.6

print('Loading data...')
(X_train, y_train), (X_Test, y_Test) = imdb.load_data(num_words=max_features)
X_test = X_Test[:int(test_ratio * len(X_Test))]
X_validation = X_Test[int(test_ratio * len(X_Test)):]
y_test = y_Test[:int(test_ratio * len(y_Test))]
y_validation = y_Test[int(test_ratio * len(y_Test)):]
print(len(X_train), 'train sequences')
print(len(X_validation), 'validation sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_validation = sequence.pad_sequences(X_validation, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_validation shape:', X_validation.shape)
print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(
    Embedding(max_features, embedding_dims, input_length=maxlen, dropout=0.2))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(
    Conv1D(
        strides=1,
        padding="valid",
        activation="relu",
        kernel_size=3,
        filters=250))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

optimizer = SGD(lr=0.02, momentum=0.1, decay=0.0, nesterov=False)

model.compile(
    loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='model_cnn.png')

model_fname = 'model_cnn.hdf5'
early_stopping = EarlyStopping(
    monitor='val_loss', min_delta=0.01, patience=10, verbose=0, mode='auto')
model_checkpoint = ModelCheckpoint(
    model_fname,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    period=1)
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    verbose=1,
    factor=0.5,
    patience=5,
    cooldown=3,
    min_lr=1e-8)
callbacks = [early_stopping, model_checkpoint, reduce_lr]

model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    validation_data=(X_validation, y_validation),
    callbacks=callbacks)

model.load_weights(model_fname)

#model_files = sorted(glob.glob('*.hdf5'))
#for model_file in model_files:
#  with h5py.File(model_file, 'a') as f:
#    if 'optimizer_weights' in f.keys():
#      del f['optimizer_weights']

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
