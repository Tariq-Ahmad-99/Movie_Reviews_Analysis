# Movie Sentiment Classifier

This project implements a movie review sentiment classifier using a **LSTM-based neural network**. It classifies reviews into two categories: **positive** or **negative**. The dataset used is the **IMDB Dataset** of movie reviews.

---

## **Table of Contents**

1. [Overview](#overview)
2. [Technologies Used](#technologies-used)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Model Testing and Web Application](#model-testing-and-web-application)
6. [Project Files](#project-files)
7. [Acknowledgments](#acknowledgments)

---

## **Overview**

- The project trains an LSTM-based deep learning model using the IMDB dataset to classify the sentiment of movie reviews.
- Tokenization of reviews is handled using **Keras' Tokenizer**, with sequences padded to ensure a uniform input length.
- A **Gradio-based web interface** is provided to make predictions on custom input reviews.

---

## **Technologies Used**

- Python 
- TensorFlow 
- Keras
- Scikit-learn
- Pandas
- Gradio
- Joblib
- NumPy
- IMDB Dataset (CSV file)

---

## **Setup Instructions**


### 1. Install Dependencies
Run the following command to install all necessary Python packages:
```bash
pip install -r requirements.txt
```

#### **Note**: Create a `requirements.txt` file with:
```text
tensorflow
keras
pandas
scikit-learn
joblib
gradio
```

### 2. Prepare the Dataset
Place the `IMDB_Dataset.csv` file inside a folder named `data`:
```plaintext
project-root/
│── data/
│   └── IMDB_Dataset.csv
```

### 3. Train the Model
Run the following script:
```bash
python train_model.py
```

This will:
- Train the model using the IMDB dataset.
- Save the trained model in the `model` folder as `model.h5`.
- Save the tokenizer as `tokenizer.pkl`.

---

## **Usage**

### **Run the Gradio Web Application**
To start the Gradio-based web app, execute:
```bash
python app.py
```

A **Gradio interface** will open in the browser. Use the text input to test the model with your custom reviews.

---

## **Model Testing and Web Application**

- The project includes a function `predictive_system` that accepts a movie review as input and predicts its sentiment.
- Example:
```python
review = "The movie was fantastic! Great story and acting."
result = predictive_system(review)
print(result)  # Output: "positive"
```

- Use the Gradio app for testing:
  - Input: Textbox for movie reviews.
  - Output: Predicted sentiment ("positive" or "negative").
  - Accessible via a web link generated by Gradio.

---

## **Project Files**

```
project-root/
│── data/
│   └── IMDB_Dataset.csv      # IMDB dataset file
│
│── model/
│   ├── model.h5              # Saved trained model
│   └── tokenizer.pkl         # Tokenizer object
│
│── train_model.py            # Script to train the model
│── app.py                    # Gradio web app
│── README.md                 # Project documentation
```

---

## **Acknowledgments**

- IMDB Dataset
- Libraries: TensorFlow, Keras, Gradio, Scikit-learn
- Gradio for providing an easy-to-use web interface.

---

## **License**

This project is licensed under the MIT License.