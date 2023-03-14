from flask import Flask, request, render_template, url_for, redirect
# import joblib
# from sklearn.externals import joblib
import pickle
import score
# import pickle

app = Flask(__name__)

fname = open("mlp",'rb')
bnb =pickle.load(fname)

threshold=0.6


@app.route('/') 
def home():
    return render_template('spam.html')


@app.route('/spam', methods=['POST'])
def spam():
    sent = request.form['sent']
    label,prop=score.score(sent,bnb,threshold)
    labl="Spam" if label == 1 else "not a spam"
    answ = f"""The sentence "{sent}" is {labl} with propensity {prop}."""
    return render_template('res.html', ans=answ)


if __name__ == '__main__': 
    app.run(debug=True)