from flask import Flask, render_template, flash, redirect
from flask.ext.wtf import Form
from wtforms import StringField, SelectField, SubmitField, TextAreaField, RadioField, BooleanField, validators
from wtforms.validators import Required
import pickle
from utils import *
from paths import *
from model import Model
from company import Company


app = Flask(__name__)
# To avoid the CSRF attck, set secret key
app.config['SECRET_KEY'] = 'Set key'


class skillForm(Form):
    skill = StringField('', validators=[Required()])
    submit = SubmitField('Submit', id="submit_button1")

class titleForm(Form):
    title = StringField('', validators=[Required()])
    submit = SubmitField('Submit', id="submit_button2")


class descriptionForm(Form):
    description = TextAreaField('', validators=[Required()])
    options = RadioField('', choices=[('s','Skills'),('t','Titles'),('j', 'Job Description')])
    submit = SubmitField('Submit', id="submit_button3")

class companyForm(Form):
    companyName = TextAreaField('', validators=[Required()])
    checkbox = BooleanField('Is a company name?', default="checked")
    submit = SubmitField('Recommend Companies', id="submit_button4")

## ------------------------------


def train():
    model = Model()
    model.train()
    pickle.dump(model, open(SAVE_MODEL_PATH, "wb"))
    company = Company()
    company.train()
    company.save()

def load():
    model = pickle.load(open(SAVE_MODEL_PATH, "rb"))
    # company = pickle.load(open(SAVE_COMPANY_MODEL_PATH, "rb"))
    company = None
    magpie = load_magpie(labels=model.skills_to_select)
    return model, company, magpie

# train()
model, company, magpie = load()
# generate_sk_categories(model.job_profiles)


## ------------------------------
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/skills2skills', methods=['GET', 'POST'])
def skills2skills():
    form = skillForm()
    result = None
    input_text = None
    # form.choice.choices = [(s,s) for s in sorted(model.skills_to_select)]

    if form.validate_on_submit():
        input_text = str(form.skill.data)
        # result = model.recommend_skills_from_skill(skill)
        skills = input_text.split(',')
        skills = [s.strip().lower() for s in skills]
        result = model.recommend_skills_from_skills(skills)
        result = [[str(k), int(v)] for k,v in result]
        # print(result)

    return render_template('skills2skills.html', result=result, form=form, input_text=input_text)


@app.route('/titles2titles', methods=['GET', 'POST'])
def titles2titles():
    form = titleForm()
    result = None
    input_text = None

    if form.validate_on_submit():
        input_text = str(form.title.data)
        titles = input_text.split(',')
        titles = [t.strip().lower() for t in titles]
        result = model.recommend_titles_from_titles(titles)
        result = [[str(k), float(v)] for k,v in result]

    return render_template('titles2titles.html', result=result, form=form)

@app.route('/title2skills', methods=['GET', 'POST'])
def title2skills():
    form = titleForm()
    result = None

    if form.validate_on_submit():
        title = form.title.data
        result = model.recommend_skills_from_title(title)
        result = [[str(k), int(v)] for k,v in result]

    return render_template('title2skills.html', result=result, form=form)

@app.route('/recommendTalents', methods=['GET', 'POST'])
def recommendTalents():
    form = descriptionForm()
    result = None
    talents = []
    required_skills = []
    option = ""

    if form.validate_on_submit():
        input_text = form.description.data
        option = form.options.data

        if not option or option == "" or option == 'j':
            result = magpie.predict_from_text(input_text)
            result = result[:20]
            required_skills = [k for k,v in result]
            
        elif option == 's':
            required_skills = parse_to_skills(input_text)

        elif option == 't':
            titles = parse_to_titles(input_text)
            required_skills = model.recommend_skills_from_titles(titles)
            required_skills = [r[0] for r in required_skills]
        else:
            pass

        talents = model.recommend_talents_from_skills(required_skills)

    return render_template('recommendTalents.html', result=result, option=option, 
        required_skills=required_skills, talents=talents, form=form)

@app.route('/companyInfo', methods=['GET', 'POST'])
def companyInfo():

    form = companyForm()
    result = None
    isCompany = True

    if form.validate_on_submit():
        input_text = form.companyName.data
        isCompany = form.checkbox.data

        if isCompany:
            result = company.get_most_similar_companies(input_text)
        else:
            result = company.recommend_companies_from_text(input_text)

    return render_template('companyInfo.html', result=result, form=form)

if __name__ == '__main__':
    app.run(debug=False)
