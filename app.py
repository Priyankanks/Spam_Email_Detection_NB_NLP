from flask import Flask, render_template, request, url_for
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['Post'])
def predict():
    df = pd.read_csv('spam.csv', encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1, inplace = True)
    df['label'] = df['class'].map({'ham': 0,'spam': 1})
    X = df['message']
    y = df['label']
    cv = CountVectorizer()
    X = cv.fit_transform(X)    
    
    X_train,X_test,y_train,y_test =  train_test_split(X,y,test_size =  0.3, random_state = 42)
    model = MultinomialNB()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    X_score= model.score(X_train,y_train)
    y_score = model.score(X_test,y_test)
    acc_score = accuracy_score(y_test,y_pred)
    print(acc_score)
    
    if request.method == 'Post':
        message = request.form['message']
        data = [message]
        vectorizor = cv.transform(data).toarray()
        pred= model.pred(vectorizor)
    return render_template('result.html', prediction = pred)

if __name__ == '__main__':
    app.run(debug = True)