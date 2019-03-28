import pickle

from gensim.models import Word2Vec
from numpy import append

cutoff = 1.96

with open("corpora/political/resplit/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [s.rstrip().split() for s in f.readlines()]
with open("corpora/political/resplit/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [s.rstrip().split() for s in f.readlines()]
with open("out/uninformed_z_scores.pickle", "rb") as f:
    uninformed_z_scores = dict([(w, z) for w, z in pickle.load(f) if abs(z) >= cutoff])

data = dem_train + rep_train
model_base = Word2Vec(sentences=data, min_count=1)
model = Word2Vec(sentences=data, min_count=1, size=101)
assert len(model_base.wv.vocab) == 89131  # Constant extracted on previous analysis.

for w in model.wv.vocab:
    if w not in uninformed_z_scores:
        model.wv.syn0[model.wv.vocab[w].index] = append(model_base.wv.syn0[model_base.wv.vocab[w].index], [0])
    else:
        model.wv.syn0[model.wv.vocab[w].index] = append(model_base.wv.syn0[model_base.wv.vocab[w].index], [1])

model.clear_sims()

model.save("out/augmented_word_embeddings.model")
