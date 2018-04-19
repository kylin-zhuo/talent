import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain
from magpie import Magpie
import pickle
from paths import *
import math


def count_cos_similarity(vec_1, vec_2):
    if len(vec_1) != len(vec_2):
        return 0

    s = sum(vec_1[i] * vec_2[i] for i in range(len(vec_2)))
    den1 = math.sqrt(sum([pow(number, 2) for number in vec_1]))
    den2 = math.sqrt(sum([pow(number, 2) for number in vec_2]))
    return s / (den1 * den2)


# Method to compute Jaccard similarity index between two sets that have different lengths
# Since sklearn.metrics.jaccard_similarity_score expects two input vectors of equal
def jaccard_similarity(arr1, arr2):
    intersection = set(arr1) & set(arr2)
    union = set(arr1) | set(arr2)
    if not union:
        return 0.0
    return len(intersection) / float(len(union))


def parse_to_skills(input_text):
    # the most basic version: just split with comma
    skills = input_text.split(',')
    skills = [str(s).strip().lower() for s in skills]
    return skills

def parse_to_titles(input_text):
    titles = input_text.split(',')
    titles = [str(s).strip().lower() for s in titles]
    return titles

# def train_model():
#     model = Model()
#     model.train()
#     pickle.dump(mode, open(SAVE_MODEL_PATH, "wb"))
#     print("Trained and persisted to")


def train_magpie(labels):
    magpie = Magpie()
    magpie.init_word_vectors(WRITE_SK_CAT_PATH, vec_dim=VEC_DIM)
    magpie.train(WRITE_SK_CAT_PATH, labels, test_ratio=0.2, epochs=EPOCHS)
    return magpie


def save_magpie(m):
    m.save_word2vec_model(SAVE_MAGPIE_WORD2VEC_PATH, overwrite=True)
    m.save_scaler(SAVE_MAGPIE_SCALER_PATH, overwrite=True)
    m.save_model(SAVE_MAGPIE_MODEL_PATH)


def load_magpie(labels):
    magpie = Magpie(
        keras_model=SAVE_MAGPIE_MODEL_PATH,
        word2vec_model=SAVE_MAGPIE_WORD2VEC_PATH,
        scaler=SAVE_MAGPIE_SCALER_PATH,
        labels=labels
    )
    return magpie

def parse_skill_string(string, excludings=['\\n', '\\N']):
    try:
        skills = string.split('\xc2\xb7')
        skills = chain(*[s.split('&') for s in skills])
        skills = [s.strip() for s in skills]
        skills = [s for s in skills if s not in excludings]
        return skills
    except:
        return []

def get_skills_from_job_descriptions(jobs, excludings=['\\n', '\\N']):
    skills = []
    bad_line_count = 0
    for i in range(len(jobs)):
        try:
            # temp = map(lambda x:x.strip().lower(), jobs.iloc[i]['skills'].split('\xc2\xb7'))
            temp = parse_skill_string(jobs.iloc[i]['skills'])
            skills.append(temp)
        except:
            bad_line_count += 1
            continue
            # print(i, jobs.iloc[i]['skills'])
    print("Number of bad lines: %d/%d" % (bad_line_count, len(jobs)))
    ret = Counter(chain(*skills))
    for w in excludings:
        if w in ret:
            ret.pop(w)
            print("Excluded: %s" % str(w))
    return ret
