from keras.layers import Input, Embedding, Reshape, Conv2D, MaxPool2D, Concatenate, Flatten, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam

from classifier_load_data import sequence_length, vocab_size, dimensions, load_word_embeddings, x_train, y_train, \
    x_val, y_val, save_outputs

# Building architecture.
filters = 512
filter_sizes = [3, 4, 5]
inputs = Input(shape=(sequence_length,), dtype="int32")
embedding = Embedding(vocab_size, dimensions, weights=[load_word_embeddings()], input_length=sequence_length,
                      trainable=False)(inputs)
reshape = Reshape((sequence_length, dimensions, 1))(embedding)
conv_0 = Conv2D(filters, kernel_size=(filter_sizes[0], dimensions), kernel_initializer="normal",
                activation="relu")(reshape)
conv_1 = Conv2D(filters, kernel_size=(filter_sizes[1], dimensions), kernel_initializer="normal",
                activation="relu")(reshape)
conv_2 = Conv2D(filters, kernel_size=(filter_sizes[2], dimensions), kernel_initializer="normal",
                activation="relu")(reshape)
maxpool_0 = MaxPool2D(pool_size=(sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
maxpool_1 = MaxPool2D(pool_size=(sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
maxpool_2 = MaxPool2D(pool_size=(sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)
concatenated = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
flatten = Flatten()(concatenated)
dropout = Dropout(0.5)(flatten)
output = Dense(1, activation="sigmoid")(dropout)
model = Model(inputs=inputs, outputs=output)
adam = Adam(lr=1e-4, epsilon=1e-08)
model.compile(optimizer=adam, loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# Running training.
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, batch_size=30)

# Saving completed model, training history, and graphical architecture.
save_outputs(model, history, "Kim", __file__)
