from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from math import ceil

model_path = "out/augmented_classifier_trainable_embed.h5"

# Loading data.
with open("corpora/political/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [s.rstrip().split() for s in f.readlines()]
with open("corpora/political/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [s.rstrip().split() for s in f.readlines()]
with open("corpora/political/sanitised/dem_test.txt", encoding="utf-8") as f:
    dem_test = [s.rstrip().split() for s in f.readlines()]
with open("corpora/political/sanitised/rep_test.txt", encoding="utf-8") as f:
    rep_test = [s.rstrip().split() for s in f.readlines()]
with open("corpora/political/sanitised/dem_dev.txt", encoding="utf-8") as f:
    dem_dev = [s.rstrip().split() for s in f.readlines()]
with open("corpora/political/sanitised/rep_dev.txt", encoding="utf-8") as f:
    rep_dev = [s.rstrip().split() for s in f.readlines()]

# Calculating max sentence length.
multiple = 50.0
sequence_length = int(ceil(max(len(s) for s in dem_train + rep_train + dem_test + rep_test) / multiple) * multiple)

# Establishing vocab indices.
tokenizer = Tokenizer()
tokenizer.fit_on_texts(dem_train + rep_train + dem_test + rep_test + dem_dev + rep_dev)

# Establishing padded samples.
dem_dev_samples = pad_sequences(tokenizer.texts_to_sequences(dem_dev), maxlen=sequence_length)
rep_dev_samples = pad_sequences(tokenizer.texts_to_sequences(rep_dev), maxlen=sequence_length)

# Loading model and formatting predictions.
model = load_model(model_path)
dem_dev_predictions = [p[0] for p in model.predict(x=dem_dev_samples).tolist()]
rep_dev_predictions = [p[0] for p in model.predict(x=rep_dev_samples).tolist()]

# Calculating overall accuracy.
correct = 0
for p in dem_dev_predictions:
    if p > 0.5:
        correct += 1
dem_acc = correct / len(dem_dev_predictions)

correct = 0
for p in rep_dev_predictions:
    if p < 0.5:
        correct += 1
rep_acc = correct / len(rep_dev_predictions)

# Writing results to file.
with open(model_path.split(".h5")[0] + "_accuracy.txt", "w") as f:
    f.write("Democrats: {}\nRepublicans: {}\nAverage: {}\n".format(dem_acc, rep_acc, (dem_acc + rep_acc) / 2))
