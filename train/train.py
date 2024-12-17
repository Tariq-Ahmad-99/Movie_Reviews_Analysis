import os
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
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

### save the model ###
folder_name = 'model'

# Create the folder if it doesn't exist
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

# Save the model
model.save(os.path.join(folder_name, 'model.h5'))
joblib.dump(tokenizer, "model/tokenizer.pkl")

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(X_test, Y_test)

print("Test Loss:", loss)

print("Test Accuracy:", accuracy)