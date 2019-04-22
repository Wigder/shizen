from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

epochs = 20

with open("corpora/resplit/sanitised/dem_test.txt", encoding="utf-8") as f:
    dem_test = f.read().split("\n")
with open("corpora/resplit/sanitised/rep_test.txt", encoding="utf-8") as f:
    rep_test = f.read().split("\n")

dem_test = [TaggedDocument(s.split(), [i]) for i, s in enumerate(dem_test) if s != ""]
rep_test = [TaggedDocument(s.split(), [i]) for i, s in enumerate(rep_test) if s != ""]

dem_model = Doc2Vec(vector_size=300)
dem_model.build_vocab(dem_test)
dem_model.train(dem_test, total_examples=dem_model.corpus_count, epochs=10)
dem_model.save("out/d2v_dem.doc2vec")

rep_model = Doc2Vec(vector_size=300)
rep_model.build_vocab(rep_test)
rep_model.train(rep_model, total_examples=rep_model.corpus_count, epochs=10)
rep_model.save("out/d2v_rep.doc2vec")
