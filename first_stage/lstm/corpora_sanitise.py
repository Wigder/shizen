from string import punctuation

from langid import classify, set_languages
from sklearn.model_selection import train_test_split

with open("corpora/custom/unsanitised/data.en", encoding="utf-8") as f:
    data_en = f.read().split("\n")
with open("corpora/custom/unsanitised/data.fr", encoding="utf-8") as f:
    data_fr = f.read().split("\n")

assert len(data_en) == len(data_fr)
data = list(zip(data_en, data_fr))

# Removing pairs with identical elements, for example: ("sentence A", "sentence A")
print("Removing pairs with identical elements...")
data = [(en, fr) for en, fr in data if en != fr]

# Removing duplicate parallel sentences.
print("Removing duplicate parallel sentences...")
data = list(set(data))

# Removing repeating source sentences aligned to multiple target sentences and repeating target sentences aligned to
# multiple source sentences.
print("Removing repeating source/target sentences...")
new_data = []
for en, fr in data:
    current_en = [en for en, _ in new_data]
    current_fr = [fr for _, fr in new_data]
    if en not in current_en and fr not in current_fr:
        new_data.append((en, fr))
data = new_data

# Removing sentences that are not in the specified source or target language.
print("Removing sentences that are not in the specified source or target language...")
set_languages(["en", "fr"])
data = [(en, fr) for en, fr in data if classify(en)[0] == "en" and classify(fr)[0] == "fr"]

# Removing punctuation and converting to lowercase.
print("Removing punctuation and converting to lowercase...")
trans = str.maketrans('', '', punctuation)
data = [(en.lower().translate(trans), fr.lower().translate(trans)) for en, fr in data]

# Splitting data into train and test sets.
print("Splitting data into train and test sets...")
train, test = train_test_split(data, test_size=1000)

# Writing to disk.
print("Writing to disk...")
with open("corpora/custom/sanitised/train.en", "w", encoding="utf-8") as f:
    for s in [en for en, _ in train]:
        f.write("{}\n".format(s))
with open("corpora/custom/sanitised/train.fr", "w", encoding="utf-8") as f:
    for s in [fr for _, fr in train]:
        f.write("{}\n".format(s))
with open("corpora/custom/sanitised/test.en", "w", encoding="utf-8") as f:
    for s in [en for en, _ in test]:
        f.write("{}\n".format(s))
with open("corpora/custom/sanitised/test.fr", "w", encoding="utf-8") as f:
    for s in [fr for _, fr in test]:
        f.write("{}\n".format(s))
