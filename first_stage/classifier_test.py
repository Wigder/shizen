from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

from classifier_load_data import data, sequence_length

# Trained model in HDF5 format.
model_path = ""

# Loading test data.
with open("corpora/political/resplit/sanitised/dem_test.txt", encoding="utf-8") as f:
    dem_test = [s.rstrip() for s in f.readlines()]
with open("corpora/political/resplit/sanitised/rep_test.txt", encoding="utf-8") as f:
    rep_test = [s.rstrip() for s in f.readlines()]

# Establishing vocab indices.
tokenizer = Tokenizer()
tokenizer.fit_on_texts([s for s, _ in data] + dem_test + rep_test)

# Establishing padded samples.
dem_test_samples = pad_sequences(tokenizer.texts_to_sequences(dem_test), maxlen=sequence_length)
rep_test_samples = pad_sequences(tokenizer.texts_to_sequences(rep_test), maxlen=sequence_length)

# Loading model and running predictions.
model = load_model(model_path)
dem_test_predictions = [p[0] for p in model.predict(x=dem_test_samples).tolist()]
rep_test_predictions = [p[0] for p in model.predict(x=rep_test_samples).tolist()]

# Calculating overall accuracy.
correct = 0
for p in dem_test_predictions:
    if p > 0.5:
        correct += 1
dem_acc = correct / len(dem_test_predictions)
correct = 0
for p in rep_test_predictions:
    if p < 0.5:
        correct += 1
rep_acc = correct / len(rep_test_predictions)

# Writing results to file.
with open(model_path.split(".h5")[0] + "_accuracy.txt", "w") as f:
    f.write("Democrats: {}\nRepublicans: {}\nAverage: {}\n".format(dem_acc, rep_acc, (dem_acc + rep_acc) / 2))
