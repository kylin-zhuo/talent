import pandas as pd
from paths import *
import scipy as sp  
import numpy as np   
from sklearn.feature_extraction.text import TfidfVectorizer
import math
import pickle

class Company(object):
	"""docstring for Company"""
	def __init__(self):
		super(Company, self).__init__()

	def read(self):
		self.comp = pd.read_csv(PATH_COMPANIES)
		# np.sum(comp.isnull())
		self.comp.dropna(subset=['company_name', 'company_description'], inplace=True)
		# remove duplicated company names 
		self.comp.drop_duplicates(subset=['company_name'], inplace=True)
		# index the company

	def generate_index(self):
		self.comp2index = dict([(str(self.comp.iloc[i]['company_name']), i) for i in range(len(self.comp))])
		self.index2comp = dict([(i, str(self.comp.iloc[i]['company_name'])) for i in range(len(self.comp))])

	def compute(self):
		tfidf_vec = TfidfVectorizer()
		comp_descriptions = [self.comp.iloc[i]['company_description'] for i in range(len(self.comp))]
		self.company_vectors = tfidf_vec.fit_transform(comp_descriptions)
		print("Company TF-IDF vectors generated.")
		self.tfidf_vec = tfidf_vec

	def get_most_similar_companies(self, target, k=20):
		if target not in self.comp2index:
			# raise Exception("The company is not in the database.")
			return []
		else:
			i = self.comp2index[target]
			vec = self.company_vectors.toarray()[i]
			sims = self.company_vectors.dot(vec)
			idxs = sims.argsort()[-k-1:-1]
			return [[str(self.index2comp[i]), sims[i]] for i in idxs[::-1]]

	def recommend_companies_from_text(self, text, k=20):
		vec = self.tfidf_vec.transform([text]).toarray().T
		sims = self.company_vectors.dot(vec).ravel()
		idxs = sims.argsort()[-k-1:-1]
		return [[str(self.index2comp[i]), sims[i]] for i in idxs[::-1]]
		
	def train(self):
		self.read()
		self.generate_index()
		self.compute()
		self.comp = None
		print("Company model trained.")

	def save(self):
		pickle.dump(self, open(SAVE_COMPANY_MODEL_PATH, "wb"))
		print("Company model saved to %s." % SAVE_COMPANY_MODEL_PATH)


