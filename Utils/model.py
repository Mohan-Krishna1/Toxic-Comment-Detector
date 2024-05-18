import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import nltk
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pickle

data = pd.read_csv("data/Dataset.csv")

data = data.drop("Unnamed: 0", axis=1)

wordnet_lemmatizer = WordNetLemmatizer()

def prepare_text(text):
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.split()
    text = ' '.join(text)
    text = word_tokenize(text)
    text = pos_tag(text)
    lemma = []
    for i in text: lemma.append(wordnet_lemmatizer.lemmatize(i[0], pos = get_wordnet_pos(i[1])))
    lemma = ' '.join(lemma)
    return lemma

data['clean_tweets'] = data['tweet'].apply(lambda x: prepare_text(x))

corpus = data['clean_tweets'].values.astype('U')

stopwords = set(nltk_stopwords.words('english'))

stopwords = list(stopwords)
count_tf_idf = TfidfVectorizer(stop_words=stopwords)
tf_idf = count_tf_idf.fit_transform(corpus)

pickle.dump(count_tf_idf, open("tf_idf.pkt", "wb"))

tf_idf_train, tf_idf_test, target_train, target_test = train_test_split(
    tf_idf, data['Toxicity'], test_size = 0.8, random_state= 42, shuffle=True
)

model_bayes = MultinomialNB()
model_bayes = model_bayes.fit(tf_idf_train, target_train)

y_pred_proba = model_bayes.predict_proba(tf_idf_test)[::, 1]

fpr, tpr, _ = roc_curve(target_test, y_pred_proba)

final_roc_auc = roc_auc_score(target_test, y_pred_proba)

test_text = "i love you"
test_tfidf = count_tf_idf.transform([test_text])
print(model_bayes.predict_proba(test_tfidf))
print(model_bayes.predict(test_tfidf))

pickle.dump(model_bayes, open("toxicity_model.pkt", "wb"))