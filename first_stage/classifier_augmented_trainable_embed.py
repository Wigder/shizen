from keras import Sequential
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense

from classifier_load_data import vocab_size, dimensions, embeddings, sequence_length, x_train, y_train, x_val, y_val, \
    save_outputs

# Building architecture.
model = Sequential()
model.add(Embedding(vocab_size, dimensions, weights=[embeddings], input_length=sequence_length))
model.add(Conv1D(128, 5, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Running training.
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=128)

# Saving completed model, training history, and graphical architecture.
save_outputs(model, history, __file__)
