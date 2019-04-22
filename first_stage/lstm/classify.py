from pickle import load

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

print("Loading classification model...")
model = load_model("classifier.h5")
# Place tokenizer and sequence length pickles in the root folder of this file to load properly.
# These pickles are outputted by "classifier_load_data.py" from the "classifier" folder.
with open("classifier_tokenizer.pickle", "rb") as f:
    tokenizer = load(f)
with open("classifier_sequence_length.pickle", "rb") as f:
    sequence_length = load(f)
democrat_label = 1
republican_label = 0


def classify(sentence_list):
    seq = pad_sequences(tokenizer.texts_to_sequences(sentence_list), maxlen=sequence_length)
    pred = model.predict(x=seq)

    return pred
