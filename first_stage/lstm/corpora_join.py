from random import shuffle

with open("corpora/original/commoncrawl/commoncrawl.fr-en.en", encoding="utf-8") as f:
    commoncrawl_en = f.readlines()
with open("corpora/original/commoncrawl/commoncrawl.fr-en.fr", encoding="utf-8") as f:
    commoncrawl_fr = f.readlines()
with open("corpora/original/europarl-v7/europarl-v7.fr-en.en", encoding="utf-8") as f:
    europarl_en = f.readlines()
with open("corpora/original/europarl-v7/europarl-v7.fr-en.fr", encoding="utf-8") as f:
    europarl_fr = f.readlines()
with open("corpora/original/news-commentary/News-Commentary.en-fr.en", encoding="utf-8") as f:
    news_en = f.readlines()
with open("corpora/original/news-commentary/News-Commentary.en-fr.fr", encoding="utf-8") as f:
    news_fr = f.readlines()

assert len(commoncrawl_en) == len(commoncrawl_fr)
commoncrawl = list(zip(commoncrawl_en, commoncrawl_fr))
assert len(europarl_en) == len(europarl_fr)
europarl = list(zip(europarl_en, europarl_fr))
assert len(news_en) == len(news_fr)
news = list(zip(news_en, news_fr))

data = commoncrawl + europarl + news
shuffle(data)

with open("corpora/custom/unsanitised/data.en", "w", encoding="utf-8") as f:
    for s in [en for en, _ in data]:
        f.write(s)
with open("corpora/custom/unsanitised/data.fr", "w", encoding="utf-8") as f:
    for s in [fr for _, fr in data]:
        f.write(s)
