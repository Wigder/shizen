from string import punctuation

from langid import classify, set_languages
from sklearn.model_selection import train_test_split

max_words = 50

print("Loading data...")
with open("corpora/custom/unsanitised/data.en", encoding="utf-8") as f:
    data_en = f.read().split("\n")
with open("corpora/custom/unsanitised/data.fr", encoding="utf-8") as f:
    data_fr = f.read().split("\n")

print("Zipping English and French sets together...")
assert len(data_en) == len(data_fr)
data = list(zip(data_en, data_fr))

# Removing pairs with identical elements, for example: ("sentence A", "sentence A")
print("Removing pairs with identical elements...")
data = [(en, fr) for en, fr in data if en != fr]

print("Removing duplicate parallel sentences...")
data = list(set(data))

# Removing repeating source sentences aligned to multiple target sentences and repeating target sentences aligned to
# multiple source sentences.
print("Removing repeating source/target sentences...")
data = [(k, v) for v, k in dict([(v, k) for k, v in dict(data).items()]).items()]

print("Removing sentences that are not in the specified source or target language...")
set_languages(["en", "fr"])
data = [(en, fr) for en, fr in data if classify(en)[0] == "en" and classify(fr)[0] == "fr"]

print("Removing punctuation and converting to lowercase...")
trans = str.maketrans('', '', punctuation)
data = [(en.lower().translate(trans), fr.lower().translate(trans)) for en, fr in data]

print("Removing sentences longer than {} words...".format(max_words))
data = [(en, fr) for en, fr in data if len(en.split()) <= max_words and len(fr.split()) <= max_words]

# Splitting data into train and test sets.
print("Splitting data into train and test sets...")
train, test = train_test_split(data, test_size=50000)

# Formatting analysis.
analysis = """Original corpus size: {}
Sanitised corpus size: {}
""".format(len(data_en), len(data))
print(analysis)

# Writing to disk.
print("Writing to disk...")
with open("corpora/custom/sanitised/train.en", "w", encoding="utf-8") as f:
    for s in [en for en, _ in train]:
        f.write("{}\n".format(s.strip()))
with open("corpora/custom/sanitised/train.fr", "w", encoding="utf-8") as f:
    for s in [fr for _, fr in train]:
        f.write("{}\n".format(s.strip()))
with open("corpora/custom/sanitised/test.en", "w", encoding="utf-8") as f:
    for s in [en for en, _ in test]:
        f.write("{}\n".format(s.strip()))
with open("corpora/custom/sanitised/test.fr", "w", encoding="utf-8") as f:
    for s in [fr for _, fr in test]:
        f.write("{}\n".format(s.strip()))
with open("out/sanitisation_analysis.txt", "w", encoding="utf-8") as f:
    f.write(analysis)
