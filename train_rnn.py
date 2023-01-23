from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.layers import TextVectorization
from keras.layers import Input
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd

x = pd.read_csv('data/x_processed.csv')["title"].values
ys = pd.read_csv('data/y_fields_filtered.csv')

for field in ys:
    print("Training for: ", field)
    y = pd.get_dummies(ys[field]).values

    classes_count = y[0].size

    print("Classes count is", classes_count)

    # split dataset for later verification
    x_train, x_verify, y_train, y_verify = train_test_split(x, y, test_size=0.25, random_state=100)

    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.33, random_state = 101)

    max_features = 20000
    embedding_dim = 128
    sequence_length = 500

    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorize_layer.adapt(X_train)

    model2 = Sequential()
    model2.add(Input(shape=(1,), dtype=tf.string, name='text'))
    model2.add(vectorize_layer)
    model2.add(layers.Embedding(max_features, embedding_dim))
    model2.add(layers.Bidirectional(layers.LSTM(20)))
    model2.add(layers.Dropout(0.3))
    model2.add(layers.Dense(classes_count, activation='softmax'))
    model2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint2 = ModelCheckpoint(f"models/RNN_{field}", monitor='val_accuracy', verbose=0,save_best_only=True, mode='auto', save_format="tf", period=6,save_weights_only=False)
    history = model2.fit(X_train, Y_train, epochs=10,validation_data=(X_test, Y_test),callbacks=[checkpoint2])

    # print(history.history.keys())
    # # summarize history for accuracy
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # summarize history for loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()