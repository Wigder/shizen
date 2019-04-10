import pickle
from collections import Counter

from word_embeddings import cutoff

with open("corpora/resplit/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = f.read().split("\n")
with open("corpora/resplit/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = f.read().split("\n")

vocab_size = len(dict(Counter(" ".join(dem_train + rep_train).split()).most_common()))

with open("out/z_scores_uninformed.pickle", "rb") as f:
    uninf = Counter(dict([(w, z) for w, z in pickle.load(f)])).most_common()
with open("out/z_scores_informed.pickle", "rb") as f:
    inf = Counter(dict([(w, z) for w, z in pickle.load(f)])).most_common()

uninf_partisan = [w for w, z in uninf if abs(z) > cutoff]
inf_partisan = [w for w, z in inf if abs(z) > cutoff]
assert set(uninf_partisan) == set(inf_partisan)
uninf_partisan_vocab_count = len(uninf_partisan)
inf_partisan_vocab_count = len(inf_partisan)

text = \
    """Vocabulary count: {}
Uninformed partisan vocabulary count: {}
Informed partisan vocabulary count: {}
Percentage of partisan vocabulary: {}
All of the words found in the set resultant from the uninformed method were found in the set resultant from the
informed method. That is, the only difference between the sets is that the associated z-scores differ slightly,
but the style specific lexicons are all exactly the same.


Top 50 words from uninformed calculation:

{}

{}


Top 50 words from informed calculation:

{}

{}
""".format(vocab_size,
           uninf_partisan_vocab_count,
           inf_partisan_vocab_count,
           uninf_partisan_vocab_count / vocab_size,
           uninf[-50:][::-1],
           uninf[:50],
           inf[-50:][::-1],
           inf[:50])

with open("out/notes_lexicon_extraction.txt", "w") as f:
    f.write(text)
