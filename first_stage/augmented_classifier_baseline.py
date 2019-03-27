import pickle
from random import shuffle

import numpy as np
from gensim.models import Word2Vec
from keras import Sequential
from keras.layers import Embedding, Conv1D, Dense, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from math import ceil

# Loading and labelling data.
with open("corpora/political/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [(s.rstrip().split(), 1) for s in f.readlines()]
with open("corpora/political/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [(s.rstrip().split(), 0) for s in f.readlines()]
with open("corpora/political/sanitised/dem_test.txt", encoding="utf-8") as f:
    dem_test = [(s.rstrip().split(), 1) for s in f.readlines()]
with open("corpora/political/sanitised/rep_test.txt", encoding="utf-8") as f:
    rep_test = [(s.rstrip().split(), 0) for s in f.readlines()]
data = dem_train + rep_train + dem_test + rep_test

# Calculating max sentence length.
multiple = 50.0
sequence_length = int(ceil(max(len(s) for s, _ in data) / multiple) * multiple)

# Establishing vocab indices.
tokenizer = Tokenizer()
tokenizer.fit_on_texts([s for s, _ in data])

# Establishing padded and senquecified data sets as well as their respective labels.
train = dem_train + rep_train
shuffle(train)
x_train = pad_sequences(tokenizer.texts_to_sequences([s for s, _ in train]), maxlen=sequence_length)
y_train = np.asarray([l for _, l in train])
test = dem_test + rep_test
shuffle(test)
x_test = pad_sequences(tokenizer.texts_to_sequences([s for s, _ in test]), maxlen=sequence_length)
y_test = np.asarray([l for _, l in test])

# Loading pre-trained word embeddings and assigning them to their respective generated indices.
w2v_model = Word2Vec.load("out/augmented_word_embeddings.model")
vocab_size = len(tokenizer.word_index) + 1
dimensions = 101
embeddings = np.zeros((vocab_size, dimensions))
for w, i in tokenizer.word_index.items():
    if w in w2v_model.wv.vocab:
        embeddings[i] = w2v_model.wv.syn0[w2v_model.wv.vocab[w].index]

# Building architecture.
model = Sequential()
model.add(Embedding(vocab_size, dimensions, weights=[embeddings], input_length=sequence_length, trainable=False))
model.add(Conv1D(128, 5, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=128)

model.save("out/augmented_classifier_baseline.h5")
with open("out/augmented_classifier_baseline_history.pickle", "wb") as f:
    pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
