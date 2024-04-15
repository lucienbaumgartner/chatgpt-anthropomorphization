import pickle
import re
import pandas as pd
from itertools import groupby
from collections import Counter, OrderedDict

## import annotated data
filepath = "../../output/data/sov/submissions/sov_submissions.pkl"
with (open(filepath, "rb")) as openfile:
    sov = pickle.load(openfile)

## Number of SVOs
print(len(sov))

## Read in MPVN
mpvn = pd.read_csv("../../input/mpvn/mpvn.csv")
print(mpvn)

### In the end, we might have to loop over subject vs object position
## Extract verbs from SVOs and MPVN
verb_lemmas_sov = [element["verb_lemma"] for element in sov]
mpvn_verbs = mpvn["Word"].tolist()

## MPVN coverage of unique verbs in sov SVO corpus
unique_verbs_covered = list(set(verb_lemmas_sov) & set(mpvn_verbs))
all_unique_verbs = set(verb_lemmas_sov).union(set(mpvn_verbs))
unique_coverage = len(unique_verbs_covered)/len(all_unique_verbs) * 100
print(f"MPVN unique coverage: {unique_coverage}")

## MPVN coverage of non-unique verbs in SVO corpus
non_unique_coverage = (len([element for element in verb_lemmas_sov if element in unique_verbs_covered]) / len(verb_lemmas_sov)) * 100
print(f"MPVN non-unique coverage: {non_unique_coverage}")

## Subset corpus
c_pre = [element for element in sov if element["verb_lemma"] in unique_verbs_covered]
c_post = [element for element in sov if element["verb_lemma"] not in unique_verbs_covered]
print(f"N Observations | training stage: {len(c_pre)} | predictions: {len(c_post)}")

dct = dict(Counter(verb_lemmas_sov))
dct = {k: v for k, v in sorted(dct.items(), key=lambda item: item[1], reverse=True)}
print(dct)