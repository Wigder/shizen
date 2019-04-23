from string import punctuation


def custom_sanitiser(sentence_list):
    # Removing new line symbol and emojis.
    new = [s.encode("ascii", "ignore").decode("ascii")  # Removing most emojis.
            .replace("&amp;", "&")  # Decoding ampersands.
            .replace("&lt; 3", "heartemoji")  # Decoding heart emojis.
            .replace(" &apos;", "")  # Decoding apostrophes.
            .replace("&quot;", "")  # Decoding quotes.
            .replace("&gt;", "")  # Decoding >.
            .replace("&lt;", "")  # Decoding <.
            .split() for s in sentence_list]
    for s in range(len(new)):
        new[s] = [w.translate(str.maketrans('', '', punctuation))
                  for i, w in (enumerate(new[s]))
                  if not (i == 0 and w in ["democratic", "republican"])]
    new = [" ".join(filter(None, s)) for s in new]
    list(filter(lambda s: s != "", new))

    return new


with open("corpora/resplit/unsanitised/dem_train.txt", encoding="utf-8") as f:
    dem_train = custom_sanitiser(f.read().split("\n"))
with open("corpora/resplit/unsanitised/dem_val.txt", encoding="utf-8") as f:
    dem_val = custom_sanitiser(f.read().split("\n"))
with open("corpora/resplit/unsanitised/dem_test.txt", encoding="utf-8") as f:
    dem_test = custom_sanitiser(f.read().split("\n"))
with open("corpora/resplit/unsanitised/rep_train.txt", encoding="utf-8") as f:
    rep_train = custom_sanitiser(f.read().split("\n"))
with open("corpora/resplit/unsanitised/rep_val.txt", encoding="utf-8") as f:
    rep_val = custom_sanitiser(f.read().split("\n"))
with open("corpora/resplit/unsanitised/rep_test.txt", encoding="utf-8") as f:
    rep_test = custom_sanitiser(f.read().split("\n"))

with open("corpora/resplit/sanitised/dem_train.txt", "w", encoding="utf-8") as f:
    for s in dem_train:
        f.write("{}\n".format(s.strip()))
with open("corpora/resplit/sanitised/dem_val.txt", "w", encoding="utf-8") as f:
    for s in dem_val:
        f.write("{}\n".format(s.strip()))
with open("corpora/resplit/sanitised/dem_test.txt", "w", encoding="utf-8") as f:
    for s in dem_test:
        f.write("{}\n".format(s.strip()))
with open("corpora/resplit/sanitised/rep_train.txt", "w", encoding="utf-8") as f:
    for s in rep_train:
        f.write("{}\n".format(s.strip()))
with open("corpora/resplit/sanitised/rep_val.txt", "w", encoding="utf-8") as f:
    for s in rep_val:
        f.write("{}\n".format(s.strip()))
with open("corpora/resplit/sanitised/rep_test.txt", "w", encoding="utf-8") as f:
    for s in rep_test:
        f.write("{}\n".format(s.strip()))
