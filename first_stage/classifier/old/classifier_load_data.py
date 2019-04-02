import pickle
from os.path import basename
from random import shuffle

import numpy as np
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import plot_model
from math import ceil


# Function for saving model, its history, and its graphical architecture.
def save_outputs(model, history, file_path_absolute):
    name = basename(file_path_absolute).split(".py")[0]
    model.save("out/{}.h5".format(name))
    plot_model(model, to_file="out/{}_architecture.png".format(name), show_shapes=True, show_layer_names=False)
    with open("out/{}_history.pickle".format(name), "wb") as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)


# Loading and labelling data.
with open("corpora/resplit/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [(s, 1) for s in f.read().split("\n")]
with open("corpora/resplit/sanitised/dem_val.txt", encoding="utf-8") as f:
    dem_val = [(s, 1) for s in f.read().split("\n")]
with open("corpora/resplit/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [(s, 0) for s in f.read().split("\n")]
with open("corpora/resplit/sanitised/rep_val.txt", encoding="utf-8") as f:
    rep_val = [(s, 0) for s in f.read().split("\n")]

data = dem_train + dem_val + rep_train + rep_val

# Calculating max sentence length and rounding to nearest multiple of 50.
multiple = 50.0
sequence_length = int(ceil(max([len(s.split()) for s, _ in data]) / multiple) * multiple)

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
w2v_model = Word2Vec.load("out/word_embeddings.model")
vocab_size = len(tokenizer.word_index) + 1
dimensions = 101
embeddings = np.zeros((vocab_size, dimensions))
for w, i in tokenizer.word_index.items():
    if w in w2v_model.wv.vocab:
        embeddings[i] = w2v_model.wv.syn0[w2v_model.wv.vocab[w].index]