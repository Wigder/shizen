from random import shuffle

import numpy as np
from keras.utils import Sequence

from lstm_utils import sequence_length, unk_token


class DataGenerator(Sequence):
    inputs = None
    targets = None
    vocab_inputs = None
    vocab_targets = None
    num_decoder_tokens = None

    def __init__(self, indices, batch_size, inputs=None, targets=None, vocab_inputs=None, vocab_targets=None,
                 num_decoder_tokens=None):
        self.indices = indices
        self.batch_size = batch_size
        self.steps = len(indices) / float(batch_size)  # Steps per epoch.
        assert self.steps.is_integer()
        self.steps = int(self.steps)
        if not DataGenerator.inputs:
            DataGenerator.inputs = inputs
        if not DataGenerator.targets:
            DataGenerator.targets = targets
        if not DataGenerator.vocab_inputs:
            DataGenerator.vocab_inputs = vocab_inputs
        if not DataGenerator.vocab_targets:
            DataGenerator.vocab_targets = vocab_targets
        if not DataGenerator.num_decoder_tokens:
            DataGenerator.num_decoder_tokens = num_decoder_tokens
        self.on_epoch_end()

    def on_epoch_end(self):
        # This may not be necessary as according to the documentation, data is shuffled anyway.
        shuffle(self.indices)

    def __len__(self):
        return self.steps

    def __getitem__(self, i):
        indices = self.indices[i * self.batch_size:(i + 1) * self.batch_size]
        encoder_input_data, decoder_input_data, decoder_target_data = self.__generate_data(indices)

        return [encoder_input_data, decoder_input_data], decoder_target_data

    def __generate_data(self, indices):
        encoder_input_data = np.zeros((self.batch_size, sequence_length), dtype="float32")
        decoder_input_data = np.zeros((self.batch_size, sequence_length), dtype="float32")
        decoder_target_data = np.zeros((self.batch_size, sequence_length, DataGenerator.num_decoder_tokens),
                                       dtype="float32")

        for i, index in enumerate(indices):
            for t, w in enumerate(DataGenerator.inputs[index].split()):
                try:
                    encoder_input_data[i, t] = DataGenerator.vocab_inputs[w]
                except KeyError:
                    encoder_input_data[i, t] = DataGenerator.vocab_inputs[unk_token]
            for t, w in enumerate(DataGenerator.targets[index].split()):
                try:
                    decoder_input_data[i, t] = DataGenerator.vocab_targets[w]
                except KeyError:
                    decoder_input_data[i, t] = DataGenerator.vocab_targets[unk_token]
                if t > 0:
                    try:
                        decoder_target_data[i, t - 1, DataGenerator.vocab_targets[w]] = 1.
                    except KeyError:
                        decoder_target_data[i, t - 1, DataGenerator.vocab_targets[unk_token]] = 1.

        return encoder_input_data, decoder_input_data, decoder_target_data
