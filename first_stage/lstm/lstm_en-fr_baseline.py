import pickle
from os.path import basename

import numpy as np
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from keras.utils import plot_model


def extract_chars(sentence_list):
    char_set = set()
    for s in sentence_list:
        for c in s:
            if c not in char_set:
                char_set.add(c)

    return char_set


# Function for saving model, its history, and its graphical architecture.
def save_outputs(model, history, encoder, decoder, file_path_absolute):
    name = basename(file_path_absolute).split(".py")[0]
    model.save("out/{}.h5".format(name))
    plot_model(model, to_file="out/{}_architecture.png".format(name), show_shapes=True, show_layer_names=False)
    with open("out/{}_history.pickle".format(name), "wb") as f:
        pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
    encoder.save("out/{}_encoder.h5".format(name))
    decoder.save("out/{}_decoder.h5".format(name))


with open("corpora/custom/sanitised/train.en", encoding="utf-8") as f:
    inputs = f.read().split("\n")
with open("corpora/custom/sanitised/train.fr", encoding="utf-8") as f:
    targets = ["\t" + s + "\n" for s in f.read().split("\n")]

input_chars = sorted(list(extract_chars(inputs)))
target_chars = sorted(list(extract_chars(targets)))
num_encoder_tokens = len(input_chars)
num_decoder_tokens = len(target_chars)
encoder_seq_length = max([len(s) for s in inputs])
decoder_seq_length = max([len(s) for s in targets])

# print("Number of samples:", len(inputs))
# print("Number of unique input tokens:", num_encoder_tokens)
# print("Number of unique output tokens:", num_decoder_tokens)
# print("Max sequence length for inputs:", encoder_seq_length)
# print("Max sequence length for outputs:", decoder_seq_length)

input_token_index = dict([(c, i) for i, c in enumerate(input_chars)])
target_token_index = dict([(c, i) for i, c in enumerate(target_chars)])

encoder_input_data = np.zeros((len(inputs), encoder_seq_length, num_encoder_tokens), dtype="float32")
decoder_input_data = np.zeros((len(inputs), decoder_seq_length, num_decoder_tokens), dtype="float32")
decoder_target_data = np.zeros((len(inputs), decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (input_text, target_text) in enumerate(zip(inputs, targets)):
    for t, c in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[c]] = 1.
    for t, c in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[c]] = 1.
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[c]] = 1.

latent_dim = 256

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy")
history = model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=64, epochs=100,
                    validation_split=0.2)

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

save_outputs(model, history, encoder_model, decoder_model, __file__)
