import pickle
from collections import Counter

with open("out/uninformed_z_scores.pickle", "rb") as f:
    uninf = Counter(dict([(w, z) for w, z in pickle.load(f)])).most_common()
with open("out/informed_z_scores.pickle", "rb") as f:
    inf = Counter(dict([(w, z) for w, z in pickle.load(f)])).most_common()

with open("out/lexicon_top_samples.txt", "w") as f:
    f.write("{}\n\n{}\n\n{}\n\n\n\n{}\n\n{}\n\n{}\n".
            format("Top 50 words from uninformed calculation:",
                   uninf[-50:][::-1],
                   uninf[:50],
                   "Top 50 words from informed calculation:",
                   inf[-50:][::-1],
                   inf[:50]))
