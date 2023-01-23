from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
from keras.layers import TextVectorization
from keras.layers import Input
from keras import regularizers
import tensorflow as tf
from sklearn.model_selection import train_test_split

import pandas as pd

x = pd.read_csv('data/x_processed.csv')["title"].values
ys = pd.read_csv('data/y_fields_filtered.csv')

for field in ys.columns:
    print("Training for: ", field)
    y = pd.get_dummies(ys[field]).values

    classes_count = y[0].size

    print("Classes count is", classes_count)

    # split dataset for later verification
    x_train, x_verify, y_train, y_verify = train_test_split(x, y, test_size=0.25, random_state=100)

    X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.25, random_state = 101)

    # values obtained by analyzing the dataset
    max_features = 8743
    sequence_length = 50
    embedding_dim = 20

    vectorize_layer = TextVectorization(
        max_tokens=max_features,
        output_mode="int",
        output_sequence_length=sequence_length,
    )
    vectorize_layer.adapt(x)

    model2 = Sequential()
    model2.add(Input(shape=(1,), dtype=tf.string, name='text'))
    model2.add(vectorize_layer)
    model2.add(layers.Embedding(max_features, embedding_dim))
    model2.add(layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)))
    model2.add(layers.MaxPooling1D(5))
    model2.add(layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)))
    model2.add(layers.GlobalMaxPooling1D())
    model2.add(layers.Dense(classes_count, activation='softmax'))
    model2.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics=['accuracy'])
    checkpoint2 = ModelCheckpoint(f"models/RNN_{field}", monitor='val_accuracy', verbose=0,save_best_only=True, mode='auto', save_format="tf", period=10,save_weights_only=False)
    history = model2.fit(X_train, Y_train, epochs=250,validation_data=(X_test, Y_test),callbacks=[checkpoint2])