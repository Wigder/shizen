from gensim.models import Doc2Vec

from doc2vec import td


# Transfers sentence style.
def transfer_style(sentence, model, doc):
    sample = sentence.split()
    similar_sentences = model.docvecs.most_similar([model.infer_vector(sample, alpha=0.025, epochs=20)])
    similar_sentences = [doc[p[0]].words for p in similar_sentences]
    new = ""
    for s in similar_sentences:
        if 0.5 < float(len(s)) / len(sample) < 2.0 and 0.5 < float(len(" ".join(s))) / len(" ".join(sample)) < 2.0:
            new = "".join(s)
            break

    return new


# Setting up experiment variables.
sentences = ["電車 が 来 て います",
             "明日 は 無料 です か",
             "お 名前 は 何 です か",
             "バス停 は どこ です か",
             "明日 は 無料 です か",
             "私 は その 店 に 行き ます",
             "それで 全部 です"]

# Transferring style and confirming with classifier.
d2v = Doc2Vec.load("out/d2v.doc2vec")
transferred = [transfer_style(s, d2v, td) for s in sentences]

# Writing up results.
template = "Original: {}\nNew: {}\n\n"
results = ""
for i, original in enumerate(sentences):
    results += template.format("".join(original.split()), transferred[i])
results = results.rstrip() + "\n"

print(results)

with open("out/results.txt", "w") as f:
    f.write(results)
