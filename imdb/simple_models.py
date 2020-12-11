import spacy
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

path = 'data/{}.csv'
train_path = path.format('fact_train')
test_path = path.format('aug_test')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

y_train = train_df.label
y_test = test_df.label

# load english module
nlp = spacy.load('en')

# remove stopwords, keep alphanumeric, tokenize
x_train_text = [' '.join([token.lemma_ for token in nlp(doc)
                          if not token.is_stop and token.is_alpha])
                for doc in tqdm(train_df.text)]

x_test_text = [' '.join([token.lemma_ for token in nlp(doc)
               if not token.is_stop and token.is_alpha])
              for doc in tqdm(test_df.text)]


# embed using tf-idf
counter = CountVectorizer()
transformer = TfidfTransformer()
x_train_counts = counter.fit_transform(x_train_text)
x_train_tfidf = transformer.fit_transform(x_train_counts)

x_test_counts = counter.transform(x_test_text)
x_test_tfidf = transformer.transform(x_test_counts)

# train and test simple models
print('Default Naive Bayes:')
clf = MultinomialNB().fit(x_train_tfidf, y_train)
y_pred = clf.predict(x_test_tfidf)
print(classification_report(y_test, y_pred))

print('Default Random Forest:')
rf_clf = RandomForestClassifier(random_state=123).fit(x_train_tfidf, y_train)
rf_y_pred = rf_clf.predict(x_test_tfidf)
print(classification_report(y_test, rf_y_pred))
