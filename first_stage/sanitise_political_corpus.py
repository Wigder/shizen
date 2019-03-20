import json
from string import punctuation


# def pretty_print(list_of_sentences):
#     print(json.dumps(list_of_sentences, indent=2))


def custom_sanitiser(sentence_list):
    # Removing new line symbol and emojis.
    new = [s.rstrip().encode("ascii", "ignore").decode("ascii")  # Removing most emojis.
            .replace("&amp;", "&")  # Decoding ampersands.
            .replace("&lt; 3", "heartemoji")  # Decoding heart emojis.
            .replace(" &apos;", "")  # Decoding apostrophes.
            .replace("&quot;", "")  # Decoding quotes.
            .replace("&gt;", "")  # Decoding >.
            .replace("&lt;", "")  # Decoding <.
            .split() for s in sentence_list]
    for s in range(len(new)):
        new[s] = [word.translate(str.maketrans('', '', punctuation)) for index, word in (enumerate(new[s]))
                  if index != 0]  # Removing punctuation and "republican"/"democrat" label.
    new = [" ".join(filter(None, s)) for s in new]
    list(filter(lambda s: s != "", new))

    return new


with open("corpora/political/modified/dem_train.txt", encoding="utf-8") as f:
    dem_train = custom_sanitiser(f.readlines())
with open("corpora/political/modified/dem_test.txt", encoding="utf-8") as f:
    dem_test = custom_sanitiser(f.readlines())
with open("corpora/political/modified/dem_dev.txt", encoding="utf-8") as f:
    dem_dev = custom_sanitiser(f.readlines())
with open("corpora/political/modified/rep_train.txt", encoding="utf-8") as f:
    rep_train = custom_sanitiser(f.readlines())
with open("corpora/political/modified/rep_test.txt", encoding="utf-8") as f:
    rep_test = custom_sanitiser(f.readlines())
with open("corpora/political/modified/rep_dev.txt", encoding="utf-8") as f:
    rep_dev = custom_sanitiser(f.readlines())

with open("corpora/political/sanitised/dem_train.txt", "w") as f:
    for s in dem_train:
        f.write("{}\n".format(s))
with open("corpora/political/sanitised/dem_test.txt", "w") as f:
    for s in dem_test:
        f.write("{}\n".format(s))
with open("corpora/political/sanitised/dem_dev.txt", "w") as f:
    for s in dem_dev:
        f.write("{}\n".format(s))
with open("corpora/political/sanitised/rep_train.txt", "w") as f:
    for s in rep_train:
        f.write("{}\n".format(s))
with open("corpora/political/sanitised/rep_test.txt", "w") as f:
    for s in rep_test:
        f.write("{}\n".format(s))
with open("corpora/political/sanitised/rep_dev.txt", "w") as f:
    for s in rep_dev:
        f.write("{}\n".format(s))
