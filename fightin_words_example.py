# This snippet is an example on how to use the fightin_words library.
# The library was originally donwloaded from
#   https://github.com/kenlimmj/fightin-words.
# It was then adapted to support Python 3.
# The algorithm in the library has been based off of the work found in Monroe et al., 2008.
import fightin_words.fightin_words as fw
import sklearn.feature_extraction.text as sk_text

# Strings/text corpora to be compared
l1 = 'The quick brown fox jumps over the lazy pig'
l2 = 'The lazy purple pig jumps over the lazier donkey'

# Extractor configuration parameters
prior = 0.05
cv = sk_text.CountVectorizer(max_features=15000)

print(fw.FWExtractor(prior, cv).fit_transform([l1, l2]))
