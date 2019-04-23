import tensorflow as tf
from keras import Model
from keras.backend import eval
from keras.callbacks import Callback
from keras.layers import Input, Embedding, Bidirectional, LSTM, Concatenate, Dense
from keras.losses import categorical_crossentropy
from numpy import mean

from classify import democrat_label, classify
from lstm_generator import DataGenerator
from lstm_load_data import load_corpora, load_data, split_indices, start_token, end_token
from lstm_utils import latent_dim, save_outputs

data_size = 220000  # 220000 is approximately 5% of all data.
y_data_size = 20000  # 20000 complements 220000 nicely.
x_batch_size = 20  # 20 can be a good choice.
y_batch_size = 2  # 2 can be a good choice.
use_multiprocessing = True  # Turn on to guarantee GPU is the only training bottleneck (and not CPU).
style_label = democrat_label
classification_loss = 0

targets, inputs = load_corpora(size=data_size)
targets, vocab_inputs, vocab_targets, num_encoder_tokens, num_decoder_tokens = load_data(inputs, targets)
x, y = split_indices(range(len(inputs)), targets, y_data_size)
# Batch size is manually calculated as factors of data length for convenience.
x_generator = DataGenerator(x, x_batch_size, inputs, targets, vocab_inputs, vocab_targets, num_decoder_tokens)
y_generator = DataGenerator(y, y_batch_size)  # Other arguments should be picked up from above.
reverse_vocab_targets = {i: w for w, i in vocab_targets.items()}


class CollectPredictions(Callback):
    def __init__(self, label):
        super(CollectPredictions, self).__init__()
        # Targets and predictions are calculated on a batch basis.
        self.pred = []
        self.label = label
        self.var_y_pred = tf.Variable(0., validate_shape=False)

    def on_batch_end(self, batch, logs=None):
        self.pred = eval(self.var_y_pred)
        # print("Predictions: ", self.pred)  # By uncommenting this line we can see strange live predictions.
        global classification_loss
        if self.pred.size != 0:
            sentences = []
            for s in self.pred[0]:
                print(s)
                sentence = ""
                for i in s:
                    if reverse_vocab_targets[i] == start_token.rstrip():
                        continue
                    elif reverse_vocab_targets[i] == end_token.lstrip():
                        break
                    sentence += reverse_vocab_targets[i] + " "
                sentences.append(sentence.rstrip())
            classification_loss = abs(self.label - mean(classify(sentences)))
        else:
            classification_loss = 0


def loss_wrapper(balancing=15):
    def generative_loss(y_true, y_pred):
        global classification_loss
        # As described by Prabhumoye et al., 2018.
        return categorical_crossentropy(y_true, y_pred) + (balancing * classification_loss)

    return generative_loss


# Define an input sequence and process it.
print("Building encoder...")
encoder_inputs = Input(shape=(None,))
encoder_emb = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder = Bidirectional(LSTM(latent_dim, return_state=True))
_, forward_h, forward_c, backward_h, backward_c = encoder(encoder_emb)  # Discarding outputs.
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
encoder_states = [state_h, state_c]

# Set up the decoder, using encoder_states as initial state.
print("Building decoder...")
decoder_inputs = Input(shape=(None,))
decoder_emb = Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder = LSTM(latent_dim * 2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(decoder_emb, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation="softmax")
decoder_outputs = decoder_dense(decoder_outputs)

print("Fitting to model...")
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer="rmsprop", loss=loss_wrapper(), metrics=["accuracy"])
model.summary()

# Setting up callback.
callback = CollectPredictions(style_label)
fetches = [tf.assign(callback.var_y_pred, model.outputs[0], validate_shape=False)]
model._function_kwargs = {"fetches": fetches}

if use_multiprocessing:
    history = model.fit_generator(generator=x_generator, validation_data=y_generator, use_multiprocessing=True,
                                  workers=4, max_queue_size=4, callbacks=[callback])
else:
    history = model.fit_generator(generator=x_generator, validation_data=y_generator, callbacks=[callback])

# Sampling models.
print("Setting up sampling models...")
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim * 2,))
decoder_state_input_c = Input(shape=(latent_dim * 2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(decoder_emb, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

print("Saving...")
save_outputs(model, encoder_model, decoder_model, history, "Generator (FR-EN)", __file__)
