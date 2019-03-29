import pickle
from random import shuffle

import numpy as np
from keras import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
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

# Building architecture.
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 101, input_length=sequence_length))
model.add(Conv1D(128, 5, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Running training.
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)

# Saving completed model and training history.
model.save("out/classifier_baseline.h5")
with open("out/classifier_baseline_history.pickle", "wb") as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
