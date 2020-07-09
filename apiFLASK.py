from flask import Flask, render_template, jsonify, request
import sklearn 
import json
import pickle as p 
import requests
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
import re

app  = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/question_pred", methods = ['POST'])
def predictques():
    data = request.get_json()
    data = data[0]
    transformed_text = vectorizer.transform(data)
    pred = np.array2string(rf.predict(transformed_text))
    
    return jsonify(pred)

@app.route("/question_type", methods = ['POST'])
def question_type():
    url = "http://localhost:5000/question_pred"


    Question = request.form["question"].rstrip()
    Question = re.sub('[^a-zA-z0-9\s]','',Question.lower())
    data = [[Question]]

    j_data = json.dumps(data)

    headers = {'content-type':'application/json','Accept-Charset':'UTF-8'}
    r = requests.post(url, data = j_data, headers = headers)
    r1 = r.text


    stat = 'Your Question is of type ' +  r1[2:-3]
 
        
    return render_template("result.html", result = stat)

if __name__ == '__main__':


    #loading x
    x = p.load(open('./saved-data/random-forest/x.pickle','rb'))
    #stemming
    stemmer = SnowballStemmer('english').stem
    def stem_tokenize(text):
        return [stemmer(i) for i in word_tokenize(text)]

    #vectorizer   
    vectorizer_file = './saved-data/random-forest/vectorizer.pickle'
    vectorizer = p.load(open(vectorizer_file,'rb'))
    x  = vectorizer.fit_transform(x.values)

    rf_file ='./saved-data/random-forest/random_forest.pickle'
    rf = p.load(open(rf_file,'rb'))

    app.run(debug = True, host = '0.0.0.0')

    