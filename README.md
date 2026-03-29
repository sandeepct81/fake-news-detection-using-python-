# 📰 Fake News Detection system Using Python 

This project is a machine learning-based solution to detect fake news articles using natural language processing techniques. It uses a dataset from Kaggle and Python-based libraries to build a binary classifier that predicts whether a news article is real or fake.

## 📌 Features

- Download dataset directly via Kaggle API
- Text preprocessing and cleaning (stopwords removal, stemming, etc.)
- Vectorization (TF-IDF or CountVectorizer)
- Model training (Logistic Regression, Naive Bayes, etc.)
- Model evaluation (accuracy, precision, recall, F1-score)
- Easy-to-run and modular code. 

## 📂 Project Structure
fake-news-detector/
│
├── data/ # Data folder (Kaggle data will be downloaded here)
├── notebooks/ # Jupyter notebooks for exploration and testing
├── src/ # Source code (preprocessing, training, etc.)
│ ├── data_loader.py
│ ├── preprocess.py
│ └── model.py
├── .kaggle/ # Contains kaggle.json API key
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── main.py # Main script to run the model 
