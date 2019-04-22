from random import shuffle

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from math import ceil

# Loading and labelling data.
with open("corpora/resplit/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [(s, 1) for s in f.read().split("\n")]
with open("corpora/resplit/sanitised/dem_val.txt", encoding="utf-8") as f:
    dem_val = [(s, 1) for s in f.read().split("\n")]
with open("corpora/resplit/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [(s, 0) for s in f.read().split("\n")]
with open("corpora/resplit/sanitised/rep_val.txt", encoding="utf-8") as f:
    rep_val = [(s, 0) for s in f.read().split("\n")]

data = [s for s, _ in dem_train + dem_val + rep_train + rep_val]

# Calculating max sentence length and rounding to nearest multiple of 50.
multiple = 50.0
sequence_length = int(ceil(max([len(s.split()) for s in data]) / multiple) * multiple)

# Establishing vocab indices.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)

# Establishing padded and sequencified data sets as well as their respective labels.
train = dem_train + rep_train
shuffle(train)
x_train = pad_sequences(tokenizer.texts_to_sequences([s for s, _ in train]), maxlen=sequence_length)
y_train = np.asarray([l for _, l in train])
val = dem_val + rep_val
shuffle(val)
x_val = pad_sequences(tokenizer.texts_to_sequences([s for s, _ in val]), maxlen=sequence_length)
y_val = np.asarray([l for _, l in val])

# Function for loading pre-trained word embeddings and assigning them to their respective generated indices.
vocab_size = len(tokenizer.word_index) + 1
dimensions = 302
