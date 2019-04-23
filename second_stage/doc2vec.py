from random import shuffle

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm


# Generates a TaggedDocument object given a sentence list.
def tagged_document(sentence_list):
    return [TaggedDocument(s.split(), [i]) for i, s in enumerate(sentence_list) if s != ""]


# Function that trains a doc2vec model given a TaggedDocument object.
def train_doc2vec(doc):
    epochs = 20
    alpha_val = 0.025
    min_alpha_val = 1e-4
    alpha_delta = (alpha_val - min_alpha_val) / (epochs - 1)
    model = Doc2Vec(vector_size=300, workers=4)
    model.build_vocab(doc)
    for _ in tqdm(range(epochs)):
        shuffle(doc)
        model.alpha, model.min_alpha = alpha_val, alpha_val
        model.train(doc, total_examples=model.corpus_count, epochs=1)
        alpha_val -= alpha_delta

    return model


with open("corpora/tokenised/ja", encoding="utf-8") as f:
    corpus = f.read().split("\n")

td = tagged_document(corpus)

if __name__ == "__main__":
    md = train_doc2vec(td)
    md.save("out/d2v.doc2vec")
