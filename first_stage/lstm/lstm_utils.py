import pickle
from os.path import basename

from keras.utils import plot_model

from lstm_plot_helper import plot_history

latent_dim = 500  # Hidden dimension of 500 as per Prabhumoye et al., 2018.
# Input size of 50 to save memory (Prabhumoye et al., 2018 uses 300).
# We add 5 extra words to it because of inputs with start and end tokens.
sequence_length = 50 + 5
unk_token = "<unk>"


# Function for saving model (and its components), its history, and its graphical architecture.
def save_outputs(model, encoder_model, decoder_model, history, title, file_path):
    name = basename(file_path).split(".py")[0]
    model.save("out/{}.h5".format(name), include_optimizer=False)
    encoder_name = "{}_encoder".format(name)
    encoder_model.save("out/{}.h5".format(encoder_name), include_optimizer=False)
    decoder_name = "{}_decoder".format(name)
    decoder_model.save("out/{}.h5".format(decoder_name), include_optimizer=False)
    history_path = "out/{}_history.pickle".format(name)
    with open(history_path, "wb") as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    plot_history(title, history_path)
    plot_model(model, to_file="out/{}_architecture.png".format(name), show_shapes=True, show_layer_names=False)
    plot_model(encoder_model, to_file="out/{}_architecture.png".format(encoder_name), show_shapes=True,
               show_layer_names=False)
    plot_model(decoder_model, to_file="out/{}_architecture.png".format(decoder_name), show_shapes=True,
               show_layer_names=False)
