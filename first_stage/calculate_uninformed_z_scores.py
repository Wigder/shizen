import pickle

from fightin_words import weighted_log_odds_dirichlet as fw
from sklearn.feature_extraction.text import CountVectorizer

with open("corpora/political/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [s.rstrip() for s in f.readlines()]
with open("corpora/political/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [s.rstrip() for s in f.readlines()]

out = fw(dem_train, rep_train, prior=.05, cv=CountVectorizer())

with open("out/uninformed_z_scores.pickle", "wb") as f:
    pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
