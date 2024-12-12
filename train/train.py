import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.layers import Dense, Embedding, LSTM # type: ignore
from keras.utils import pad_sequences # type: ignore


# Load the dataset from a CSV file
data = pd.read_csv("data/IMDB_Dataset.csv")

# label encoding
data.replace({"sentiment": {"positive" : 1 , "negative" : 0 }}, inplace = True)

# Split the data
train_data, test_data = train_test_split( data, test_size = 0.2, random_state = 42)

# Tokenize the training dataset
tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(train_data["review"])

# but the data in pad sequences
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen = 200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen = 200)

Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]


# Model creation	
model = Sequential()
model.add(Embedding(input_dim = 5000, output_dim = 128, input_length = 200))
model.add(LSTM(128, dropout = 0.2, recurrent_dropout = 0.2))
model.add(Dense(1, activation = "sigmoid"))


# Model training
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
model.fit(X_train, Y_train, epochs = 5, batch_size = 64, validation_split = 0.2)