import os
from random import shuffle

from natto import MeCab
from tqdm import tqdm

# Setting up MeCab env vars for the custom MeCab 64 bit installation.
os.environ["MECAB_PATH"] = "C:\\Program Files\\MeCab\\bin\\libmecab.dll"
os.environ["MECAB_CHARSET"] = "utf-8"

with open("corpora/original/jesc/ja", encoding="utf-8") as f:
    jesc = f.read().split("\n")
with open("corpora/original/opensubs/ja", encoding="utf-8") as f:
    opensubs = f.read().split("\n")

custom = list(set(jesc + opensubs))
shuffle(custom)
new = ""

with MeCab(r"-F%m,%f[0]") as mecab:
    for s in tqdm(custom):
        tokens = []
        for n in mecab.parse(s, as_nodes=True):
            if n.is_nor():
                pos = n.feature.split(",")
                # Removes punctuation and unnecessary symbols hopefully.
                # Indices below will only work if MeCab has been declared as MeCab(r"-F%m,%f[0]").
                if pos[1] != "記号":
                    tokens.append(pos[0])
        tokens = " ".join(tokens)
        if tokens != "":
            new += tokens + "\n"

with open("corpora/tokenised/ja", "w", encoding="utf-8") as f:
    f.write(new)
