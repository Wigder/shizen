import pickle
from collections import Counter

from fightin_words import weighted_log_odds_dirichlet as fw
from sklearn.feature_extraction.text import CountVectorizer

common_words = ["the", "to", "and", "you", "of", "for", "a", "is", "in", "i", "that", "are", "your", "this", "we",
                "not", "it", "have", "on", "with", "be", "our", "will", "as", "do", "no", "so", "but", "what", "from",
                "if", "us", "has", "they", "who", "would", "by", "or", "my", "was", "can", "dont", "at", "should", "an",
                "one", "when", "why", "their", "how", "its", "there", "am", "go", "them", "been", "im", "were",
                "because", "then", "me", "than", "did", "any", "had", "does"]
"""
Common words were handpicked from the top 500 common words.
"""

with open("corpora/political/sanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = [s.rstrip() for s in f.readlines()]
with open("corpora/political/sanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = [s.rstrip() for s in f.readlines()]

word_count = dict(Counter(" ".join(dem_train + rep_train).split()).most_common())

priors = {w: 1.0 / word_count[w] for w in common_words}
for w in word_count:
    if w not in priors:
        priors[w] = .05

prior = []
cv_vocab = {}
for w, p in priors.items():
    prior.append(p)
    cv_vocab[w] = len(prior) - 1

out = fw(dem_train, rep_train, prior=prior, cv=CountVectorizer(vocabulary=cv_vocab))

with open("out/informed_z_scores.pickle", "wb") as f:
    pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
