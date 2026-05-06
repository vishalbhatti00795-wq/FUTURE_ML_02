# FUTURE_ML_02

# 🎫 Support Ticket Classification System

A Machine Learning and Natural Language Processing (NLP) based application developed to automatically classify customer support tickets into different categories. This project was built as part of my internship at Future Interns.

---

# 📌 Project Overview

Customer support teams receive a large number of tickets daily. Manually categorizing these tickets can be time-consuming and inefficient.

This project uses:

* Natural Language Processing (NLP)
* TF-IDF Vectorization
* Machine Learning Algorithms
* Streamlit GUI

to automatically classify support tickets into predefined categories and determine ticket priority.

---

# 🚀 Features

✅ Automatic Support Ticket Classification
✅ Text Cleaning and Preprocessing
✅ TF-IDF Feature Extraction
✅ Machine Learning Model Training
✅ Ticket Priority Detection
✅ Confusion Matrix Visualization
✅ Interactive Streamlit GUI
✅ Real-time Ticket Prediction

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib
* Seaborn
* NLP (Natural Language Processing)

---

# ⚙️ Machine Learning Workflow

The project follows the complete Machine Learning pipeline:

1. Data Collection
2. Text Cleaning
3. Feature Extraction using TF-IDF
4. Train-Test Split
5. Model Training using Logistic Regression
6. Model Evaluation
7. Prediction System
8. GUI Deployment using Streamlit

---

# 🧹 Text Preprocessing

The text preprocessing steps include:

* Converting text to lowercase
* Removing special characters
* Removing stopwords
* Cleaning user input text

Example:

```python id="57ggic"
"Issue with Login!!!"
```

becomes:

```python id="48tl06"
issue with login
```

---

# 🤖 Machine Learning Model

The project uses:

```python id="nrbj2z"
LogisticRegression()
```

for support ticket classification.

TF-IDF Vectorization is used to convert text data into numerical features for machine learning.

---

# 📊 Evaluation Metrics

The model is evaluated using:

* Accuracy Score
* Classification Report
* Confusion Matrix

---

# 🎯 Ticket Priority Detection

The system also predicts ticket priority levels:

* 🔴 High Priority
* 🟠 Medium Priority
* 🟢 Low Priority

based on specific keywords in the support ticket text.

---

# 🖥️ Streamlit GUI

The project includes an interactive Streamlit interface where users can:

* Enter support ticket text
* Predict ticket category
* View ticket priority
* View model evaluation graphs

---

# ▶️ How to Run the Project

## 1️⃣ Clone Repository

```bash id="p6k5s0"
git clone https://github.com/your-username/Support-Ticket-Classification-System.git
```

---


## 3️⃣ Run Streamlit App

```bash id="3n00cb"
streamlit run app.py
```

---

# 📷 Output Screens

The application displays:

* Dataset Preview
* Accuracy Score
* Confusion Matrix
* Ticket Category Prediction
* Ticket Priority

---

# 📈 Future Improvements

* Use larger datasets for better accuracy
* Deploy the project on cloud platforms
* Add Deep Learning models
* Integrate chatbot support
* Add database connectivity

---

# 🎓 Internship Information

This project was developed during my internship at Future Interns as part of a Machine Learning and NLP based real-world application.

---

# 👨‍💻 Author

## Vishal Bhatti

Machine Learning & AI Enthusiast 🚀

---

# 📜 License

This project is open-source and available for educational and learning purposes.
