# -----------------------------------
# SUPPORT TICKET CLASSIFICATION SYSTEM
# STREAMLIT GUI APPLICATION
# -----------------------------------

# Import Libraries
import streamlit as st
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Support Ticket Classifier",
    page_icon="🎫",
    layout="wide"
)

# -----------------------------------
# TITLE
# -----------------------------------

st.title("🎫 Support Ticket Classification System")

st.markdown("### Machine Learning based Support Ticket Classifier")

st.write(
    "This project uses Natural Language Processing (NLP) "
    "and Machine Learning to classify customer support tickets."
)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("support_tickets.csv")

# -----------------------------------
# SHOW DATASET
# -----------------------------------

st.subheader("📄 Dataset Preview")

st.dataframe(df.head())

# -----------------------------------
# TEXT CLEANING FUNCTION
# -----------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', '', text)

    return text

# Apply Cleaning
df["clean_text"] = df["text"].apply(clean_text)

# -----------------------------------
# TF-IDF FEATURE EXTRACTION
# -----------------------------------

vectorizer = TfidfVectorizer(stop_words='english')

X = vectorizer.fit_transform(df["clean_text"])

y = df["category"]

# -----------------------------------
# TRAIN TEST SPLIT
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------
# MODEL TRAINING
# -----------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -----------------------------------
# MODEL PREDICTIONS
# -----------------------------------

y_pred = model.predict(X_test)

# -----------------------------------
# ACCURACY SCORE
# -----------------------------------

accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Accuracy")

st.success(f"Accuracy: {accuracy * 100:.2f}%")

# -----------------------------------
# CLASSIFICATION REPORT
# -----------------------------------

st.subheader("📊 Classification Report")

report = classification_report(y_test, y_pred)

st.text(report)

# -----------------------------------
# CONFUSION MATRIX
# -----------------------------------

st.subheader("📉 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(8, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=model.classes_,
    yticklabels=model.classes_,
    ax=ax
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

st.pyplot(fig)

# -----------------------------------
# CATEGORY DISTRIBUTION
# -----------------------------------

st.subheader("📌 Ticket Category Distribution")

fig2, ax2 = plt.subplots(figsize=(8, 5))

df["category"].value_counts().plot(
    kind='bar',
    ax=ax2
)

plt.xlabel("Category")

plt.ylabel("Count")

plt.title("Support Ticket Categories")

st.pyplot(fig2)

# -----------------------------------
# PRIORITY FUNCTION
# -----------------------------------

def get_priority(text):

    text = text.lower()

    high_keywords = [
        "urgent",
        "immediately",
        "critical",
        "server down",
        "not working",
        "asap"
    ]

    medium_keywords = [
        "issue",
        "problem",
        "slow",
        "delay"
    ]

    for word in high_keywords:

        if word in text:

            return "High Priority 🔴"

    for word in medium_keywords:

        if word in text:

            return "Medium Priority 🟠"

    return "Low Priority 🟢"

# -----------------------------------
# USER INPUT SECTION
# -----------------------------------

st.subheader("📝 Predict Support Ticket Category")

user_input = st.text_area(
    "Enter Customer Support Ticket",
    height=150
)

# -----------------------------------
# PREDICTION BUTTON
# -----------------------------------

if st.button("Predict Category"):

    if user_input.strip() == "":

        st.warning("Please enter support ticket text.")

    else:

        # Clean Input
        cleaned_input = clean_text(user_input)

        # Vectorize Input
        vector_input = vectorizer.transform([cleaned_input])

        # Predict Category
        prediction = model.predict(vector_input)[0]

        # Get Priority
        priority = get_priority(user_input)

        # Display Result
        st.subheader("🎯 Prediction Result")

        st.success(f"Predicted Category: {prediction}")

        st.info(f"Ticket Priority: {priority}")

# -----------------------------------
# FOOTER
# -----------------------------------

st.markdown("---")

st.markdown(
    "Developed using Python, Streamlit, NLP, TF-IDF and Machine Learning 🚀"
)