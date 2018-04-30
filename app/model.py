import numpy as np
import pandas as pd
import json
from itertools import combinations, permutations, chain
from collections import Counter, defaultdict
import pickle
from magpie import Magpie
from utils import *
from paths import *


class Model(object):

    def __init__(self):
        self.paths_talents = [PATH_TALENT1, PATH_TALENT2]
        self.path_job = PATH_JOBS
        self.talent_skill_profiles = []
        self.job_profiles = None
        self.n_bad_records = 0
        self.n_candidates = 0
        self.skill_counter = None
        self.skills = []
        self.skill_cooc = defaultdict(dict) # to dump
        self.skills_to_select = []
        self.titles_to_select = []
        self.title_skills = None # to dump

    def get_short_profile(self, js):
        # self-defined method for extacting partial information from a talent profile
        basic = js['basic']
        uid = str(basic['id'])
        fullname = basic['fullname']
        age = basic['age']
        email = basic['email']
        skills = [str(s).lower().strip() for s in basic['skill']] 
        image_url = basic['image_url']
        linkedin_url = basic['linkedin_url']
        location = basic['location']
        education = js['education']
        return {'basic': basic, 'skills': skills, 'education': education}

    def read_talent_skill_profiles(self):
        self.n_candidates = 0
        for path in self.paths_talents:
            with open(path, 'rU') as f:
                for line in f:
                    js = json.loads(line, encoding="utf8")
                    dic = dict(js)
                    try:
                        # sk = dic['basic']['skill']
                        # sk = [str(s).lower().strip() for s in sk] 
                        # uid = str(dic['basic']['id'])
                        # fullname = str(dic['basic']['fullname'])
                        # self.talent_skill_profiles.append((uid, fullname, sk))
                        self.talent_skill_profiles.append(self.get_short_profile(dic))
                        self.n_candidates += 1
                    except:
                        self.n_bad_records += 1
                        continue
        print("Extracted %d candidates' skills. Bad records: %d" % (self.n_candidates, self.n_bad_records))
        self.skill_profiles = [u['skills'] for u in self.talent_skill_profiles]
        print("Length of skill profiles: %d" % len(self.skill_profiles))

    def read_job_profiles(self):
        self.job_profiles = pd.read_csv(self.path_job, error_bad_lines=False, header=None)
        self.job_profiles.columns = ['id', 'title', 'skills', 'description']
        self.job_profiles.dropna(subset=['title', 'description'], inplace=True)
        self.job_profiles['title'] = self.job_profiles['title'].apply(lambda x:x.strip().strip('\\').lower())

    # def save_skill_profiles(self, path="../data/skill_profiles.pkl"):
    #     pickle.dump(self.skill_profiles, open(path, "wb" ))
    #     print("Skill profiles saved to %s." % path)

    # def load_skill_profiles(self, path="../data/skill_profiles.pkl"):
    #     self.skill_profiles = pickle.load(open(path, "rb"))
    #     print("Skill profiles loaded from %s." % path)

    def get_skill_counter(self):
        self.skill_counter = Counter(chain(*self.skill_profiles))
        print("All (%d) skills extracted." % len(self.skill_counter))

    def filter_skill_counter(self, threshold = 20):
        self.skill_counter = dict((k,v) for k,v in self.skill_counter.iteritems() if v > threshold)
        print("Filtered out the skills appearing not more than %d times." % threshold)
        print("%d skills remaining." % len(self.skill_counter))

    def get_skills(self):
        self.skills = [k for k,v in sorted(self.skill_counter.items(), key=lambda x:-x[1])]
        print("Stored the skills.")

    # This solves the problem 1 by calculating the cooccurrence.
    def calculate_skill_cooc(self):
        for sp in self.skill_profiles:
            for left, right in permutations(sorted(sp), 2):
                self.skill_cooc[left][right] = self.skill_cooc[left].get(right, 0) + 1
        print("Skill cooccurrence calculated.")

    # def save_skill_cooc(self, path="../data/skill_cooc.pkl"):
    #     if self.skill_cooc:
    #         pickle.dump(self.skill_cooc, open(path, "wb"))
    #         print("Skill cooccurrence saved in %s." % path)
    #     else:
    #         print("No skill cooccurrence data.")

    # # Shortcut - load the calculated data.
    # def load_skill_cooc(self, path="../data/skill_cooc.pkl"):
    #     self.skill_cooc = pickle.load(open(path, "rb"))
    #     print("Skill cooccurrence loaded from %s." % path)

    def train(self):
        self.read_talent_skill_profiles()
        self.read_job_profiles()
        self.calculate_skill_cooc()
        self.calculate_title_skills()
        self.get_skills_to_select()
        self.get_titles_to_select()
        self.get_skill_counter()
        self.read_job_profiles()
        self.calculate_title_skills()

    # def load_model(self):
    #     self.load_skill_profiles()
    #     self.load_skill_cooc()
    #     self.get_skill_counter()
    #     self.read_job_profiles()
    #     self.calculate_title_skills()


    def recommend_skills_from_skill(self, input_skill, quant=20):
        """
        Recommend from single skill
        """
        input_skill = input_skill.lower()
        if input_skill not in self.skill_cooc:
            print("The input skill not found.")
            return []
        items = self.skill_cooc[input_skill].items()
        items.sort(key=lambda x:-x[1])
        return items[:quant]

    def recommend_skills_from_skills(self, input_skills, quant=20):
        """
        Recommend from a set of skills
        """
        ctr = Counter()
        for sk in input_skills:
            temp = Counter(dict(self.recommend_skills_from_skill(sk, 1000)))
            ctr.update(temp)
        items = ctr.items()
        items.sort(key=lambda x:-x[1])
        return items[:quant]

    def calculate_title_skills(self):
        self.title_skills = defaultdict(Counter)
        for i in range(len(self.job_profiles)):
            title = self.job_profiles.iloc[i]['title']
            skills = parse_skill_string(self.job_profiles.iloc[i]['skills'])
            skills_ctr = Counter(skills)
            self.title_skills[title].update(skills_ctr)

    def recommend_titles_from_title(self, input_title, quant=20):
        """
        Compute the similarities between two titles.
        The title can be expressed by its required skills. 
        Use Jaccard Similarity
        """
        if not self.title_skills:
            self.calculate_title_skills()
        if input_title not in self.title_skills:
            print("Title %s not in database." % input_title)
            return []
        ranks = [[str(t), jaccard_similarity(self.title_skills[t], self.title_skills[input_title])] for t in self.title_skills if t != input_title]
        # ranks = [[str(t), len(self.title_skills[t] & self.title_skills[input_title]) * 1. / (1+len(self.title_skills[t] | self.title_skills[input_title]))]  for t in self.title_skills if t != input_title]
        ranks.sort(key=lambda x:-x[1])
        return ranks[:quant]

    def recommend_titles_from_titles(self, input_titles, quant=20):
        ctr = Counter()
        for tt in input_titles:
            temp = Counter(dict(self.recommend_titles_from_title(tt, 1000)))
            ctr.update(temp)
        items = ctr.items()
        items.sort(key=lambda x:-x[1])
        return items[:quant]

    def recommend_skills_from_title(self, input_title, k=20):
        input_title = input_title.lower()
        if not self.title_skills[input_title]:
            print("Title %s not in the database." % input_title)
            return []
        else:
            return sorted(self.title_skills[input_title].items(), key=lambda x:-x[1])[:k]

    def recommend_skills_from_titles(self, input_titles, k=20):
        ctr = Counter()
        for tt in input_titles:
            temp = Counter(dict(self.recommend_skills_from_title(tt, 1000)))
            ctr.update(temp)
        items = ctr.items()
        items.sort(key=lambda x:-x[1])
        print(items[:k])
        return items[:k]

    def recommend_talents_from_skills(self, skills, k=20):
        skills = [s.lower() for s in skills]
        scores = []
        for i in range(len(self.talent_skill_profiles)):
            try:
                prof = self.talent_skill_profiles[i]
                sim = jaccard_similarity(skills, prof['skills'])
                # name = prof['basic']['fullname']
                # scores.append([name, sim])
                scores.append([prof, sim])
            except:
                print("Error in %d" % i)
        scores.sort(key=lambda x:-x[1])
        return scores[:k]


    def get_skills_to_select(self):
        skills_to_select = get_skills_from_job_descriptions(self.job_profiles)
        skills_to_select = sorted(skills_to_select.items(), key=lambda x:-x[1])
        skills_to_select = filter(lambda x:x[1] > 10, skills_to_select)
        skills_to_select = [k for k,v in skills_to_select]
        self.skills_to_select = skills_to_select

    def get_titles_to_select(self):
        titles_to_select = sorted(list(self.job_profiles['title'].unique()))

    # Generate the training set for multi-label text classification
    def generate_training_skill_categories(self):
        for i in range(len(self.job_profiles)):
            skills = parse_skill_string(self.job_profiles.iloc[i]['skills'])
            if skills:
                with open("%s%d.lab" % (WRITE_SK_CAT_PATH, i), "w+") as f:
                    f.write("\n".join(skills))
                with open("%s%d.txt" % (WRITE_SK_CAT_PATH, i), "w+") as f:
                    f.write(str(s['description']))
