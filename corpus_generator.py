from random import shuffle
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from math import floor
# import nltk
# nltk.download("punkt")
# nltk.download('stopwords')
from natto import MeCab
import os

# Setting up MeCab paths because of the unofficial 64 bit installation
os.environ["MECAB_PATH"] = "C:\Program Files\MeCab\\bin\libmecab.dll"
os.environ["MECAB_CHARSET"] = "utf-8"


def prepare(path):
    with open(path, "r", encoding="utf-8") as f:
        out = f.readlines()

    return out


jesc_en = prepare("corpora/jesc/raw/en")
jesc_ja = prepare("corpora/jesc/raw/ja")

opensubs_en = prepare("corpora/opensubs/txt/raw/en")
opensubs_ja = prepare("corpora/opensubs/txt/raw/ja")

jesc = list(zip(jesc_en, jesc_ja))
opensubs = list(zip(opensubs_en, opensubs_ja))
# print(jesc[:5])
# print(jesc[-5:])
# print(opensubs[:5])
# print(opensubs[-5:])
custom = jesc + opensubs
shuffle(custom)

en_stop = set(stopwords.words("english"))
punctuation_table = str.maketrans(dict.fromkeys(punctuation))
# https://github.com/stopwords-iso/stopwords-ja/blob/master/stopwords-ja.txt
ja_stop = {"あそこ", "あっ", "あの", "あのかた", "あの人", "あり", "あります", "ある", "あれ", "い", "いう", "います", "いる", "う",
           "うち", "え", "お", "および", "おり", "おります", "か", "かつて", "から", "が", "き", "ここ", "こちら", "こと", "この",
           "これ", "これら", "さ", "さらに", "し", "しかし", "する", "ず", "せ", "せる", "そこ", "そして", "その", "その他",
           "その後", "それ", "それぞれ", "それで", "た", "ただし", "たち", "ため", "たり", "だ", "だっ", "だれ", "つ", "て", "で",
           "でき", "できる", "です", "では", "でも", "と", "という", "といった", "とき", "ところ", "として", "とともに", "とも",
           "と共に", "どこ", "どの", "な", "ない", "なお", "なかっ", "ながら", "なく", "なっ", "など", "なに", "なら", "なり",
           "なる", "なん", "に", "において", "における", "について", "にて", "によって", "により", "による", "に対して", "に対する",
           "に関する", "の", "ので", "のみ", "は", "ば", "へ", "ほか", "ほとんど", "ほど", "ます", "また", "または", "まで", "も",
           "もの", "ものの", "や", "よう", "より", "ら", "られ", "られる", "れ", "れる", "を", "ん", "何", "及び", "彼", "彼女",
           "我々", "特に", "私", "私達", "貴方", "貴方方"}

val_len = 5000
test_len = floor(len(custom) * 0.05)
train_len = len(custom) - val_len - test_len


def custom_tokenize(pair, en_corpus, ja_corpus, mecab):
    # This function is very specific and was only made to avoid repetition in the code below where it is called 3 times
    en_no_stop = " ".join([word.translate(punctuation_table) for word in word_tokenize(pair[0].lower())
                           if word.translate(punctuation_table) not in en_stop
                           and len(word.translate(punctuation_table)) > 2])
    en_corpus.write(en_no_stop)
    en_corpus.write("\n")
    ja_tokenized = []
    for n in mecab.parse(pair[1], as_nodes=True):
        if n.is_nor():
            pos = n.feature.split(",")
            # Removes punctuation and unnecessary symbols hopefully
            # Indexes below will only work if MeCab has been declared as MeCab(r"-F%m,%f[0]")
            if pos[1] != "記号" and pos[0] not in ja_stop:
                ja_tokenized.append(pos[0])
    ja_corpus.write(" ".join(ja_tokenized))
    ja_corpus.write("\n")


with open("corpora/subs_custom/raw/en_train.txt", "w", encoding="utf-8") as en_train, \
        open("corpora/subs_custom/raw/ja_train.txt", "w", encoding="utf-8") as ja_train, \
        open("corpora/subs_custom/raw/en_val.txt", "w", encoding="utf-8") as en_val, \
        open("corpora/subs_custom/raw/ja_val.txt", "w", encoding="utf-8") as ja_val, \
        open("corpora/subs_custom/raw/en_test.txt", "w", encoding="utf-8") as en_test, \
        open("corpora/subs_custom/raw/ja_test.txt", "w", encoding="utf-8") as ja_test, \
        MeCab(r"-F%m,%f[0]") as nm:
    for pair in custom[:train_len]:
        custom_tokenize(pair, en_train, ja_train, nm)
    for pair in custom[train_len:train_len + val_len]:
        custom_tokenize(pair, en_val, ja_val, nm)
    for pair in custom[train_len + val_len:train_len + val_len + test_len]:
        custom_tokenize(pair, en_test, ja_test, nm)
