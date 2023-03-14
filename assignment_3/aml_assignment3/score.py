import numpy as np
# import pandas as pdde
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# import jobli
import pandas as pd
rawdata=pd.read_csv("rawdata.csv")
train= pd.read_csv("train.csv")
test= pd.read_csv("test.csv")
validate= pd.read_csv("validate.csv")
X_train= train.Text
Y_train= train.Label
X_validate = validate.Text
Y_validate= validate.Label
X_test = test.Text
Y_test = test.Label

cnt = CountVectorizer().fit(rawdata.Text)
X_train=cnt.transform(X_train)
X_val = cnt.transform(X_validate)
X_test = cnt.transform(X_test)

tfidf_transform = TfidfTransformer()
tfidf_train = tfidf_transform.fit_transform(X_train)
tfidf_val= tfidf_transform.fit_transform(X_val)
tfidf_test = tfidf_transform.fit_transform(X_test)

Y_train= Y_train.astype('int')
Y_validate= Y_validate.astype('int')
Y_test = Y_test.astype('int')


# from sklearn.externals import joblib
import pickle

def text_vec(text):
    obs= cnt.transform([text])
    obs = tfidf_transform.fit_transform(obs)
    return obs

fname = open("mlp",'rb')
mlp =pickle.load(fname)

def score(text:str, model, threshold:float=0.6) -> (bool,float):
    # Transform the input text using the same used during training
    eb = text_vec(text)
    print(eb.shape)
    # Predict the propensity score for the input text for each class
    prediction=model.predict(eb)
    propensity = model.predict_proba(eb)
    return prediction[0], propensity[0]

print(score("Open this Email to grab our sale freebies !!!",mlp,0.6))