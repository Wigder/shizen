import pickle

from fightin_words import weighted_log_odds_dirichlet as fw
from sklearn.feature_extraction.text import CountVectorizer

with open("corpora/resplit/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = f.read().split("\n")
with open("corpora/resplit/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = f.read().split("\n")

out = fw(dem_train, rep_train, prior=.05, cv=CountVectorizer())

with open("out/z_scores_uninformed.pickle", "wb") as f:
    pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
