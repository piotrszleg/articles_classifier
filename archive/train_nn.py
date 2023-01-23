from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
from sklearn.feature_extraction.text import CountVectorizer
from keras.layers import Input
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd

x = pd.read_csv('data/x_processed.csv')["title"].values
ys = pd.read_csv('data/y_fields_filtered.csv')

for field in ["League"]:
    print("Training for: ", field)
    y = pd.get_dummies(ys[field]).values

    classes_count = y[0].size

    print("Classes count is", classes_count)

    # split dataset for later verification
    x_train, x_verify, y_train, y_verify = train_test_split(x, y, test_size=0.25, random_state=100)

    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 101)

    vectorizer = CountVectorizer()
    vectorizer.fit(x_train)

    X_train = vectorizer.transform(X_train)
    X_test  = vectorizer.transform(X_test)

    model2 = Sequential()
    model2.add(layers.Input(shape=len(vectorizer.vocabulary_)))
    model2.add(layers.Dense(64, activation='relu'))
    model2.add(layers.Dropout(0.25))
    model2.add(layers.Dense(64, activation='relu'))
    model2.add(layers.Dropout(0.1))
    model2.add(layers.Dense(classes_count, activation='softmax'))
    model2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint2 = ModelCheckpoint(f"models/NN_{field}", monitor='val_accuracy', verbose=0,save_best_only=True, mode='auto', save_format="tf", period=6,save_weights_only=False)
    history = model2.fit(X_train, Y_train, epochs=30,validation_data=(X_test, Y_test),callbacks=[checkpoint2])

    # print(history.history.keys())
    # # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()