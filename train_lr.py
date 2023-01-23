from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import pandas as pd
import pickle

x = pd.read_csv('data/x_processed.csv')["title"].values
ys = pd.read_csv('data/y_fields_filtered.csv')

# train for each label column or field
for column in ys:
   print("Training for", column)
   y = ys[column].values

   # split dataset for later verification
   x_train, x_verify, y_train, y_verify = train_test_split(
      x, y, test_size=0.25, random_state=100)
   
   x_train, x_test, y_train, y_test = train_test_split(
      x_train, y_train, test_size=0.25, random_state=101)

   vectorizer = CountVectorizer()
   vectorizer.fit(x_train)

   X_train = vectorizer.transform(x_train)
   X_test  = vectorizer.transform(x_test)

   classifier = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=800)
   classifier.fit(X_train, y_train)
   score = classifier.score(X_test, y_test)

   print("Accuracy:", score)

   # save vectorizer
   filename = f"models/{column}.v"
   print("Saving to", filename)
   with open(filename, "wb") as f:
      pickle.dump(vectorizer, f)
   
   # save linear regression model
   filename = f"models/{column}.lrm"
   print("Saving to", filename)
   with open(filename, "wb") as f:
      pickle.dump(classifier, f)
