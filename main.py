# Model Testing & Web Application
import joblib
from keras.utils import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore
import gradio as gr

# Loading model 
tokenizer = joblib.load("model/tokenizer.pkl")
model = load_model('model/model.h5')


def predictive_system(review):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0]>0.5 else "negative"
    return sentiment

# the_result = predictive_system("i couldn't complete the movie because its too hard to understand") # to do single prediction


# Simple web app using gradio
title = "Movie Emotion Classifier"
app = gr.Interface(fn = predictive_system, inputs="textbox", outputs="textbox", title=title)
app.launch(share=True)