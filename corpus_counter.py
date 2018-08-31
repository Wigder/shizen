paths = ["corpora/subs_custom/raw/en_train.txt",
         "corpora/subs_custom/raw/ja_train.txt",
         "corpora/subs_custom/raw/en_val.txt",
         "corpora/subs_custom/raw/ja_val.txt",
         "corpora/subs_custom/raw/en_test.txt",
         "corpora/subs_custom/raw/ja_test.txt"]

for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        out = f.readlines()

    print("\"{}\":".format(path), len(out))
