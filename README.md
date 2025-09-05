Amazon Alexa Reviews – Sentiment Analysis

This project analyzes the Amazon Alexa Reviews dataset and builds machine learning models to predict whether a given review is positive or negative.

 Project Overview

Performed Exploratory Data Analysis (EDA) on Amazon Alexa product reviews.

Preprocessed review text (cleaning, stemming, stopword removal).

Converted text into numerical features using CountVectorizer (Bag of Words).

Trained and compared multiple ML models:

✅ Random Forest

✅ XGBoost

✅ Decision Tree

Evaluated models using Accuracy, Confusion Matrix, and Cross-Validation.

Features

EDA with bar plots, pie charts, histograms.

Wordclouds for positive and negative reviews.

 Text preprocessing: stemming + stopword removal.

 Machine Learning models with hyperparameter tuning (GridSearchCV).

Model saving with pickle for reuse.

Dataset

File: amazon_alexa.tsv

Columns:

rating → User’s rating (1–5)

verified_reviews → Text review

feedback → Sentiment label (0 = Negative, 1 = Positive)

variation → Alexa product type

⚠️ Dataset is not included in repo. You can download it from Kaggle – Amazon Alexa Reviews
.

Model Workflow

Load dataset and remove null values.

Preprocess reviews → cleaning, stemming, stopword removal.

Convert text into numerical features using CountVectorizer.

Train/Test split (70% train, 30% test).

Scale features with MinMaxScaler.

Train models:

Random Forest

XGBoost

Decision Tree

Evaluate with accuracy & confusion matrix.

📊 Results
Model	Training Acc.	Testing Acc.	Cross-Validation Acc.
Random Forest	~98%	~93%	~92%
XGBoost	~97%	~92%	~91%
Decision Tree	~96%	~90%	~88%

(Values may vary depending on random state & parameters)

🚀 How to Run
1️⃣ Clone the repository
git clone https://github.com/shailjam14/Amazon-Alexa-Sentiment-Analysis/
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Add dataset

Place amazon_alexa.tsv in the project root directory.

4️⃣ Run the script
python amazon_alexa.py

🛠 Tech Stack

Python

Pandas, NumPy, Matplotlib, Seaborn

NLTK

Scikit-learn

XGBoost

WordCloud

🔮 Future Work

Deploy the model using Streamlit / Flask API.

Explore deep learning models (LSTM, BERT, RoBERTa).

Improve hyperparameter tuning with Optuna.

Add real-time sentiment prediction UI.

📜 License

This project is licensed under the MIT License.


Do you also want me to make a requirements.txt for this project like I did for Twitter?
