from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score

from classifier_load_data import tokenizer, sequence_length
from googletrans import Translator

# Passing in the relative path of the trained HDF5 format model.
def calculate_accuracy(model_path):
    # Loading test data.
    with open("corpora/resplit/sanitised/dem_test.txt", encoding="utf-8") as f:
        dem_test = f.read().split("\n")
    with open("corpora/resplit/sanitised/rep_test.txt", encoding="utf-8") as f:
        rep_test = f.read().split("\n")

    # Establishing padded samples.
    dem_test_samples = pad_sequences(tokenizer.texts_to_sequences(dem_test), maxlen=sequence_length)
    rep_test_samples = pad_sequences(tokenizer.texts_to_sequences(rep_test), maxlen=sequence_length)

    # Loading model and running predictions.
    model = load_model(model_path)
    dem_test_predictions = [1 if p[0] >= 0.5 else 0 for p in model.predict(x=dem_test_samples).tolist()]
    rep_test_predictions = [0 if p[0] < 0.5 else 1 for p in model.predict(x=rep_test_samples).tolist()]

    # Calculating overall accuracy.
    dem_acc = accuracy_score([1 for _ in dem_test_predictions], dem_test_predictions)
    rep_acc = accuracy_score([0 for _ in rep_test_predictions], rep_test_predictions)

    # Printing results then writing to file.
    accuracy = "Democrats: {}\nRepublicans: {}\nAverage: {}\n".format(dem_acc, rep_acc, (dem_acc + rep_acc) / 2)
    print(accuracy)
    with open(model_path.split(".h5")[0] + "_accuracy.txt", "w") as f:
        f.write(accuracy)
