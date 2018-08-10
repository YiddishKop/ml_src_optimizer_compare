from __future__ import print_function

import glob

import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.datasets import imdb
from keras.layers import (GRU, LSTM, Activation, Dense, Dropout, Embedding,
                          SimpleRNN)
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils, plot_model

np.random.seed(42)  # for reproducibility


max_features = 20000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32
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
model.add(Embedding(max_features, 128, dropout=0.2))
model.add(LSTM(128, dropout=0.2,
               recurrent_dropout=0.2))  # try using a GRU instead, for fun
model.add(Dense(1))
model.add(Activation('sigmoid'))

optimizer = SGD(lr=0.02, momentum=0.2, decay=0.0, nesterov=False)

# try using different optimizers and different optimizer configs
model.compile(
    loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
print(model.summary())
plot_model(model, to_file='model_lstm.png')

model_fname = 'model_lstm.hdf5'
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

print('Train...')
model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=100,
    validation_data=(X_validation, y_validation),
    callbacks=callbacks)

model.load_weights(model_fname)

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)
