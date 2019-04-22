from random import shuffle

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm


def tagged_document(sentence_list):
    return [TaggedDocument(s.split(), [i]) for i, s in enumerate(sentence_list) if s != ""]


def train_doc2vec(doc):
    epochs = 20
    alpha_val = 0.025
    min_alpha_val = 1e-4
    alpha_delta = (alpha_val - min_alpha_val) / (epochs - 1)
    model = Doc2Vec(size=300, workers=4)
    model.build_vocab(doc)
    for _ in tqdm(range(epochs)):
        shuffle(doc)
        model.alpha, model.min_alpha = alpha_val, alpha_val
        model.train(doc, total_examples=model.corpus_count, epochs=1)
        alpha_val -= alpha_delta

    return model


with open("corpora/resplit/sanitised/dem_test.txt", encoding="utf-8") as f:
    dem_test = f.read().split("\n")
with open("corpora/resplit/sanitised/rep_test.txt", encoding="utf-8") as f:
    rep_test = f.read().split("\n")

dem_td = tagged_document(dem_test)
rep_td = tagged_document(rep_test)

if __name__ == "__main__":
    dem_model = train_doc2vec(dem_td)
    rep_model = train_doc2vec(rep_td)

    dem_model.save("out/d2v_dem.doc2vec")
    rep_model.save("out/d2v_rep.doc2vec")
