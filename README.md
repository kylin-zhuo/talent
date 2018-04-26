#Explanation of solution

1) Find skills from skills

Calculate the cooccurrences of each pair of skills using the tanlents' profiles, for a certain skill, recommend the skills that have high occurrence with it.

2) Find titles from titles

Extract the feature of a title using its corresponding skills. Define a similarity measure (in this approach Jaccard similarity) and pick most similar titles.

3) Find skills from title

Straightforwardly take the most frequent skills associated with a title.

4) Recommend talents from skills/titles/job description

(a) from description: Train a multi-class text classification model and predict the probabilities of skills as classes of each description text. Following step will be (c).

(b) from titles: call the function of "Find skills from title" for each titles and integrate the skills. Then follow (c).

(c) Generate the talent profiles which contain their skills. Find the most similar talents given a certain set of skills.

5) Company information

Extract the TF-IDF vectors from the description of each company as its feature. Recommend based on cosine similarity of the feature. P.S. the calculation can take seconds.

-----
#Run the program

pip install -r requirements.txt
cd app/
python app.py

It can take 10-20 seconds to load the model.

Running on localhost http://127.0.0.1:5000/ (or a diffenrent port)
