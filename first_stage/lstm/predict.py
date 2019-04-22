from keras.models import load_model
from numpy import zeros, argmax
from tqdm import tqdm

from lstm_load_data import load_corpora, load_data, start_token, end_token, unk_token
from lstm_utils import sequence_length

# These values need to be the same as the ones using during training.
data_size = 220000
y_data_size = 20000

inputs, targets = load_corpora(size=data_size)
_, vocab_inputs, vocab_targets, _, _ = load_data(inputs, targets)
reverse_vocab_inputs = {i: w for w, i in vocab_inputs.items()}
reverse_vocab_targets = {i: w for w, i in vocab_targets.items()}

with open("corpora/custom/sanitised/test.en", encoding="utf-8") as f:
    test_inputs = f.read().split("\n")
with open("corpora/custom/sanitised/test.fr", encoding="utf-8") as f:
    test_targets = f.read().split("\n")

encoder_model = load_model("out/lstm_en-fr_baseline_encoder.h5")
decoder_model = load_model("out/lstm_en-fr_baseline_decoder.h5")
predictions = []
true = []


def translate(sentence):
    input_seq = zeros(sequence_length)
    for i, w in enumerate(sentence.split()):
        try:
            input_seq[i] = vocab_inputs[w]
        except KeyError:
            input_seq[i] = vocab_inputs[unk_token]
    states = encoder_model.predict(input_seq)
    target_seq = zeros((1, 1))
    target_seq[0, 0] = vocab_targets[start_token.rstrip()]
    pred = ""
    while True:
        output, h, c = decoder_model.predict([target_seq] + states)
        sampled_index = argmax(output[0, -1, :])
        sampled = reverse_vocab_targets[sampled_index]
        if sampled == end_token.lstrip() or len(pred.split()) > sequence_length:
            break
        pred += sampled + " "
        target_seq = zeros((1, 1))
        target_seq[0, 0] = sampled_index
        states = [h, c]

    return pred.rstrip()


for i, s in tqdm(enumerate(test_inputs)):
    predictions.append(translate(s))

with open("out/predictions.txt", "w", encoding="utf-8") as f:
    for s in predictions:
        f.write("{}\n".format(s.strip()))
