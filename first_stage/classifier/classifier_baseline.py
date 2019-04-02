from keras.layers import Input, Embedding, Reshape, Conv2D, MaxPool2D, Flatten, Dropout, Dense
from keras.models import Model

from classifier_load_data import sequence_length, vocab_size, dimensions, x_train, y_train, x_val, y_val, save_outputs

# Building architecture.
kernel_size = 5
inputs = Input(shape=(sequence_length,), dtype="int32")
embedding = Embedding(vocab_size, dimensions, input_length=sequence_length)(inputs)
reshape = Reshape((sequence_length, dimensions, 1))(embedding)
conv = Conv2D(100, kernel_size=(kernel_size, dimensions), kernel_initializer="normal", activation="relu")(reshape)
maxpool = MaxPool2D(pool_size=(sequence_length - kernel_size + 1, 1), strides=(1, 1))(conv)
flatten = Flatten()(maxpool)
dropout = Dropout(0.5)(flatten)
output = Dense(1, activation="sigmoid")(dropout)
model = Model(inputs=inputs, outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Running training.
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=30)

# Saving completed model, training history, and graphical architecture.
save_outputs(model, history, __file__)
