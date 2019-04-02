import pickle
from os.path import basename

from gensim.models import Word2Vec
from numpy import append

cutoff = 1.96
embedding_dimensions = 300
lexicon_membership_dimensions = 2

with open("corpora/resplit/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [s.split() for s in f.read().split("\n")]
with open("corpora/resplit/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [s.split() for s in f.read().split("\n")]
with open("out/z_scores_uninformed.pickle", "rb") as f:
    z_scores = dict([(w, z) for w, z in pickle.load(f) if abs(z) >= cutoff])

data = dem_train + rep_train
model_base = Word2Vec(sentences=data, min_count=1, size=embedding_dimensions)
model = Word2Vec(sentences=data, min_count=1, size=embedding_dimensions + lexicon_membership_dimensions)
assert len(model_base.wv.vocab) == 89131  # Constant extracted on previous analysis.

for w in model.wv.vocab:
    if w not in z_scores:
        model.wv.syn0[model.wv.vocab[w].index] = append(model_base.wv.syn0[model_base.wv.vocab[w].index], [0, 0])
    else:
        if z_scores[w] > 0:
            # Assigning 1 to the second to last dimension of the vector if the word has democrat statistical relevance.
            model.wv.syn0[model.wv.vocab[w].index] = append(model_base.wv.syn0[model_base.wv.vocab[w].index], [1, 0])
        elif z_scores[w] < 0:
            # Assigning 1 to the last dimension of the vector if the word has republican statistical relevance.
            model.wv.syn0[model.wv.vocab[w].index] = append(model_base.wv.syn0[model_base.wv.vocab[w].index], [0, 1])
        else:
            raise ValueError

model.clear_sims()

model.save("out/{}.model".format(basename(__file__).split(".py")[0]))
