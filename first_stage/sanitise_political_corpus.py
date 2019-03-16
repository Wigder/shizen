import json
from string import punctuation


def pretty_print(list_of_sentences):
    print(json.dumps(list_of_sentences, indent=2))


def custom_sanitiser(sentence_list):
    # Removing new line symbol and emojis.
    new = [s.rstrip().encode("ascii", "ignore").decode("ascii").split() for s in sentence_list]
    for s in range(len(new)):
        new[s] = [t for t in new[s] if ((not (new[s].index(t) == 0 and t in ["democratic", "republican"]))
                                        and "&apos;" not in t  # Removing apostrophes.
                                        and not any(p in t for p in punctuation))]  # Removing punctuation.
    new = [" ".join(s) for s in new]
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
