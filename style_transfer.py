from string import punctuation

from gensim.models import Doc2Vec
from googletrans import Translator
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from classifier_load_data import tokenizer, sequence_length
from doc2vec import dem_td, rep_td


# Transforms prediction probabilities into class labels, asserting that the confidence in the predictions is strong.
def prob_to_class(probs):
    assert False not in [p <= 0.5 or p >= 0.95 for p in probs]
    return [1 if p >= 0.5 else 0 for p in probs]


# Removes punctuation from sentences.
def clean(sentence_list):
    return [" ".join(s.translate(str.maketrans(punctuation, ' ' * len(punctuation))).split()) for s in sentence_list]


# Back translates (language A -> language B -> language A).
def back_translate(sentence, transl):
    return transl.translate(transl.translate(sentence, dest="fr", src="en").text, dest="en", src="fr").text


# Transfers sentence style.
def transfer_style(sentence, model, doc):
    sample = sentence.split()
    similar_sentences = model.docvecs.most_similar([model.infer_vector(sample, alpha=0.025, epochs=20)])
    similar_sentences = [doc[p[0]].words for p in similar_sentences]
    new = ""
    for s in similar_sentences:
        if 0.5 < float(len(s)) / len(sample) < 2.0 and 0.5 < float(len(" ".join(s))) / len(" ".join(sample)) < 2.0:
            new = " ".join(s)
            break

    return new


# Setting up experiment variables.
translator = Translator()
dem_label = 1
rep_label = 0
dem_sentences = ["as a hoosier, i thank you, rep. visclosky.",
                 "thank you for standing up for justice and against bigotry"
                 "--racism, homophobia, sexism, misogyny, religious and xenophobia.",
                 "thank you for all you are doing for us, attorney general harris!"]
rep_sentences = ["i will continue praying for you and the decisions made by our government!",
                 "tom, i wish u would bring change.",
                 "all talk and no action-why dont you have some guts like breitbart"]

# Back translating sentences.
dem_back_trans = [back_translate(s, translator) for s in dem_sentences]
rep_back_trans = [back_translate(s, translator) for s in rep_sentences]

# Cleaning back translated sentences to remove punctuation.
dem_clean = clean(dem_back_trans)
rep_clean = clean(rep_back_trans)

# Formatting clean back translated sentences for classification.
dem_samples = pad_sequences(tokenizer.texts_to_sequences(dem_clean), maxlen=sequence_length)
rep_samples = pad_sequences(tokenizer.texts_to_sequences(rep_clean), maxlen=sequence_length)

# Classifying back translations and asserting that they still belong to the original style.
classifier = load_model("out/classifier_baseline.h5")
dem_samples_pred = prob_to_class([p[0] for p in classifier.predict(dem_samples).tolist()])
assert rep_label not in dem_samples_pred
rep_samples_pred = prob_to_class([p[0] for p in classifier.predict(rep_samples).tolist()])
assert dem_label not in rep_samples_pred

# Transferring style and confirming with classifier.
dem_d2v = Doc2Vec.load("out/d2v_dem.doc2vec")
rep_d2v = Doc2Vec.load("out/d2v_rep.doc2vec")
dem_to_rep = [transfer_style(s, rep_d2v, rep_td) for s in dem_clean]
rep_to_dem = [transfer_style(s, dem_d2v, dem_td) for s in rep_clean]
dem_to_rep_samples = pad_sequences(tokenizer.texts_to_sequences(dem_to_rep), maxlen=sequence_length)
rep_to_dem_samples = pad_sequences(tokenizer.texts_to_sequences(rep_to_dem), maxlen=sequence_length)
dem_to_rep_samples_pred = prob_to_class([p[0] for p in classifier.predict(dem_to_rep_samples).tolist()])
assert dem_label not in dem_to_rep_samples_pred
rep_to_dem_samples_pred = prob_to_class([p[0] for p in classifier.predict(rep_to_dem_samples).tolist()])
assert rep_label not in rep_to_dem_samples_pred

# Writing up results.
template = "Label: {}\nOriginal: {}\nDemocrat: {}\nRepublican: {}\n\n"
results = ""
for i, original in enumerate(dem_sentences):
    results += template.format(dem_label, original, dem_back_trans[i], dem_to_rep[i])
for i, original in enumerate(rep_sentences):
    results += template.format(rep_label, original, rep_to_dem[i], rep_back_trans[i])
results = results.rstrip() + "\n"

print(results)

with open("out/results.txt", "w", encoding="utf-8") as f:
    f.write(results)
