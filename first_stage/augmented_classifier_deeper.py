import pickle
from random import shuffle

import numpy as np
from gensim.models import Word2Vec
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Reshape, Flatten, Dropout, Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from math import ceil

# Loading and labelling data.
with open("corpora/political/resplit/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [(s.rstrip().split(), 1) for s in f.readlines()]
with open("corpora/political/resplit/sanitised/dem_val.txt", encoding="utf-8") as f:
    dem_val = [(s.rstrip().split(), 1) for s in f.readlines()]
with open("corpora/political/resplit/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [(s.rstrip().split(), 0) for s in f.readlines()]
with open("corpora/political/resplit/sanitised/rep_val.txt", encoding="utf-8") as f:
    rep_val = [(s.rstrip().split(), 0) for s in f.readlines()]

data = dem_train + dem_val + rep_train + rep_val

# Calculating max sentence length and rounding to nearest multiple of 50.
multiple = 50.0
sequence_length = int(ceil(max(len(s) for s, _ in data) / multiple) * multiple)

# Establishing vocab indices.
tokenizer = Tokenizer()
tokenizer.fit_on_texts([s for s, _ in data])

# Establishing padded and sequencified data sets as well as their respective labels.
train = dem_train + rep_train
shuffle(train)
x_train = pad_sequences(tokenizer.texts_to_sequences([s for s, _ in train]), maxlen=sequence_length)
y_train = np.asarray([l for _, l in train])
val = dem_val + rep_val
shuffle(val)
x_val = pad_sequences(tokenizer.texts_to_sequences([s for s, _ in val]), maxlen=sequence_length)
y_val = np.asarray([l for _, l in val])

# Loading pre-trained word embeddings and assigning them to their respective generated indices.
w2v_model = Word2Vec.load("out/augmented_word_embeddings.model")
vocab_size = len(tokenizer.word_index) + 1
dimensions = 101
embeddings = np.zeros((vocab_size, dimensions))
for w, i in tokenizer.word_index.items():
    if w in w2v_model.wv.vocab:
        embeddings[i] = w2v_model.wv.syn0[w2v_model.wv.vocab[w].index]

# Building architecture.
filters = 512
filter_sizes = [3, 4, 5]
inputs = Input(shape=(sequence_length,), dtype="int32")
embedding = Embedding(vocab_size, dimensions, weights=[embeddings], input_length=sequence_length,
                      trainable=False)(inputs)
reshape = Reshape((sequence_length, dimensions, 1))(embedding)
conv_0 = Conv2D(filters, kernel_size=(filter_sizes[0], dimensions), kernel_initializer="normal",
                activation="relu")(reshape)
conv_1 = Conv2D(filters, kernel_size=(filter_sizes[1], dimensions), kernel_initializer="normal",
                activation="relu")(reshape)
conv_2 = Conv2D(filters, kernel_size=(filter_sizes[2], dimensions), kernel_initializer="normal",
                activation="relu")(reshape)
maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)
concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated_tensor)
dropout = Dropout(0.5)(flatten)
output = Dense(units=1, activation="sigmoid")(dropout)
model = Model(inputs=inputs, outputs=output)
checkpoint = ModelCheckpoint("weights.{epoch:03d}-{val_acc:.4f}.hdf5", monitor="val_acc", verbose=1,
                             save_best_only=True)
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, batch_size=30, epochs=10, verbose=1, callbacks=[checkpoint],
                    validation_data=(x_val, y_val))

model.save("out/augmented_classifier_deeper.h5")
with open("out/augmented_classifier_deeper_history.pickle", "wb") as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
