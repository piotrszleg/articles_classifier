from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# disable keras logging 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras

x = pd.read_csv('data/x_processed.csv')["title"].values
ys = pd.read_csv('data/y_fields_filtered.csv')

for column in ys:
    print("\n#", column)
    y = ys[column].values

    x_train, x_verify, y_train, y_verify = train_test_split(x, y, test_size=0.25, random_state=100)
    
    vectorizer = pickle.load(open(f"models/{column}.v", 'rb'))
    x_verify_vectorized = vectorizer.transform(x_verify)

    base_model = pickle.load(open(f"models/{column}.lrm", 'rb'))
    print(f"Reference model on {column}")
    print(base_model.score(x_verify_vectorized, y_verify))

    rnn_model = keras.models.load_model(f'models/RNN_{column}')
    columns = pd.get_dummies(ys[column]).columns
    
    y_verify_dummies = pd.DataFrame(columns=columns, dtype="float32")

    for i, row in enumerate(y_verify):
       for field in columns:
          y_verify_dummies.at[i, field] = 0.0
       y_verify_dummies.at[i, row] = 1.0

    y_verify_dummies = y_verify_dummies.values

    print(f"RNN model on {column}")
    print(rnn_model.evaluate(x_verify, y_verify_dummies, verbose=0)[1])