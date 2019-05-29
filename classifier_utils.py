import pickle
from os.path import basename

from keras.utils import plot_model

from classifier_plot_helper import plot_history
from classifier_test import calculate_accuracy


# Function for saving model, its accuracy, its history, its training plot, and its graphical architecture.
def save_outputs(model, history, title, file_path, y_range=None):
    name = basename(file_path).split(".py")[0]
    model_path = "out/{}.h5".format(name)
    model.save(model_path)
    calculate_accuracy(model_path)
    history_path = "out/{}_history.pickle".format(name)
    with open(history_path, "wb") as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    plot_history(title, history_path, y_range)
    plot_model(model, to_file="out/{}_architecture.png".format(name), show_shapes=True, show_layer_names=False)
