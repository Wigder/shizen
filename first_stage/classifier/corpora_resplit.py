from random import shuffle

from sklearn.model_selection import train_test_split

with open("corpora/original/democratic_only.train.en", encoding="utf-8") as f:
    dem_train = f.readlines()
with open("corpora/original/democratic_only.dev.en", encoding="utf-8") as f:
    dem_val = f.readlines()
with open("corpora/original/democratic_only.test.en", encoding="utf-8") as f:
    dem_test = f.readlines()
with open("corpora/original/republican_only.train.en", encoding="utf-8") as f:
    rep_train = f.readlines()
with open("corpora/original/republican_only.dev.en", encoding="utf-8") as f:
    rep_val = f.readlines()
with open("corpora/original/republican_only.test.en", encoding="utf-8") as f:
    rep_test = f.readlines()

dem_data = dem_train + dem_val + dem_test
shuffle(dem_data)
rep_data = rep_train + rep_val + rep_test
shuffle(rep_data)

# The data has already been shuffled but we leave the shuffle argument defaulted (True) in order to add more randomness.
dem_train, dem_val = train_test_split(dem_data, test_size=0.1)
dem_val, dem_test = train_test_split(dem_val, test_size=0.5)
rep_train, rep_val = train_test_split(rep_data, test_size=0.1)
rep_val, rep_test = train_test_split(rep_val, test_size=0.5)

# print(len(dem_train)/len(dem_data))
# print(len(dem_val)/len(dem_data))
# print(len(dem_test)/len(dem_data))
# print(len(rep_train)/len(rep_data))
# print(len(rep_val)/len(rep_data))
# print(len(rep_test)/len(rep_data))

with open("corpora/resplit/unsanitised/dem_train.txt", "w", encoding="utf-8") as f:
    for s in dem_train:
        f.write(s)
with open("corpora/resplit/unsanitised/dem_val.txt", "w", encoding="utf-8") as f:
    for s in dem_val:
        f.write(s)
with open("corpora/resplit/unsanitised/dem_test.txt", "w", encoding="utf-8") as f:
    for s in dem_test:
        f.write(s)
with open("corpora/resplit/unsanitised/rep_train.txt", "w", encoding="utf-8") as f:
    for s in rep_train:
        f.write(s)
with open("corpora/resplit/unsanitised/rep_val.txt", "w", encoding="utf-8") as f:
    for s in rep_val:
        f.write(s)
with open("corpora/resplit/unsanitised/rep_test.txt", "w", encoding="utf-8") as f:
    for s in rep_test:
        f.write(s)
