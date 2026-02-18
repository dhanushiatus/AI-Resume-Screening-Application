# 📄 AI Resume Screening Application

An intelligent Resume Screening System that automates resume parsing, information extraction, and job category prediction using **Machine Learning** and **Natural Language Processing (NLP)**.  

This project combines **TF-IDF vectorization**, a **K-Nearest Neighbors (KNN) classifier**, and a modern **Streamlit** web interface to classify resumes into **25 job roles** with high accuracy.

---

## 🚀 Project Highlights

- 🔍 **Automated Resume Parsing** (PDF & TXT support)  
- 🤖 **Machine Learning-Based Job Prediction**  
- 🧠 **NLP-Powered Text Cleaning & Feature Extraction**  
- 📊 **25 Job Role Classification**  
- 📧 **Smart Extraction of Name, Email, Phone**  
- 💻 **Detection of Technical Skills & Programming Languages**  
- 🌙 **Professional Dark-Themed Streamlit UI**  
- ⚡ **Real-Time Prediction with Confidence Score**  

---

## 🧠 Machine Learning Workflow
```
Resume Text → Cleaning → TF-IDF Vectorization → KNN Model → Category Prediction
```

- **Vectorizer:** TF-IDF  
- **Classifier:** K-Nearest Neighbors (KNN)  
- **Evaluation Metric:** Accuracy Score  
- **Categories:** 25 Professional Job Roles  

---

## 🛠️ Tech Stack

**Languages & Frameworks:**  
- Python 3.8+  
- Streamlit  
- Scikit-learn  
- Jupyter Notebook  

**Libraries:**  
- Pandas & NumPy  
- NLTK  
- Matplotlib & Seaborn  
- PyPDF2  
- Regex  

---

## 📂 Project Structure

```
Resume-Screening-App/
│
├── app.py
├── model_training.ipynb
├── UpdatedResumeDataSet.csv
├── clf.pkl
├── tfidf.pkl
├── requirements.txt
└── README.md
```


---

## 🎯 Key Features

- ✔ Intelligent **Name Filtering** (removes titles & resume keywords)  
- ✔ Robust **Email & International Phone Extraction**  
- ✔ **Skill & Programming Language Detection**  
- ✔ **Professional Summary Extraction**  
- ✔ **Interactive Preview Section**  
- ✔ Modular Architecture (**Model + UI Separation**)  

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt
python -m pip install PyPDF2 

# Run the Streamlit app
streamlit run app.py
```
Upload a resume and instantly get:

Extracted personal information

Predicted job category

Confidence score

Skills & summary

📌 Future Improvements

Theme toggle (Light/Dark)

Batch resume processing

Export results (CSV/PDF)

REST API integration

Model comparison (Logistic Regression, SVM, etc.)

Database storage

📊 Project Impact

This project demonstrates strong skills in:

Machine Learning Model Development

Natural Language Processing

Data Cleaning & Feature Engineering

Model Deployment with Streamlit

End-to-End AI Application Development

It can be used by HR departments, recruitment agencies, and hiring platforms to streamline the resume screening process.


