from keras.preprocessing.text import Tokenizer
from numpy import array
from sklearn.model_selection import train_test_split

from lstm_utils import unk_token

start_token = "START_ "
end_token = " _END"
corpus_length = 4729171


# Returns a list of vocabulary given a sentence list (corpus).
def generate_vocab(sentence_list, target):
    # Removing "_" from filters and turning off lowercase conversion in order to preserve our start and end tokens.
    tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n', lower=False)
    print("Tokenising...")
    tokenizer.fit_on_texts(sentence_list)
    vocab_size = 100002  # As defined by Prabhumoye et al., 2018. Adding 2 to it due to start and end tokens.
    print("Reducing vocabulary...")
    reduced_vocab = sorted(tokenizer.word_counts.items(), key=lambda p: p[1], reverse=True)[:vocab_size]
    vocab = {w: tokenizer.word_index[w] for w, _ in reduced_vocab}
    reduced_vocab_dict = dict(reduced_vocab)
    if target:
        assert start_token.rstrip() in reduced_vocab_dict
        assert end_token.lstrip() in reduced_vocab_dict
    assert unk_token not in reduced_vocab_dict
    unk_token_i = 0
    assert unk_token_i not in [i for _, i in vocab.items()]
    vocab[unk_token] = unk_token_i
    try:
        assert len(vocab) == vocab_size + 1
    except AssertionError:
        print("WARNING: vocabulary size is not {}.".format(vocab_size + 1))

    return vocab


def load_corpora(size=corpus_length):
    print("Loading corpora...")
    with open("corpora/custom/sanitised/train.en", encoding="utf-8") as f:
        data_en = f.read().split("\n")
    with open("corpora/custom/sanitised/train.fr", encoding="utf-8") as f:
        data_fr = f.read().split("\n")

    assert corpus_length == len(data_en) == len(data_fr)

    return data_en[:size], data_fr[:size]


# Returns formatted target sentence list, vocabulary, and vocabulary count.
def load_data(inputs, targets):
    print("Adding start and end tokens to targets...")
    targets = [start_token + s + end_token for s in targets]

    vocab_inputs = generate_vocab(inputs, False)
    vocab_targets = generate_vocab(targets, True)

    print("Getting number of tokens...")
    num_encoder_tokens = len(vocab_inputs)
    num_decoder_tokens = len(vocab_targets)

    return targets, vocab_inputs, vocab_targets, num_encoder_tokens, num_decoder_tokens


# Generates a split for training and validation sets given input and target indices.
def split_indices(input_indices, target_indices, test_size):
    assert len(input_indices) == len(target_indices)
    # If using all data, a good split is 4256300-472870.
    x, y = train_test_split(input_indices, test_size=test_size, shuffle=False)
    return array(x), array(y)
