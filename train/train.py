import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, Embedding, LSTM
from keras.utils import pad_sequences




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
