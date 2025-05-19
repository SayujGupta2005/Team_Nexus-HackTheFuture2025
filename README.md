---

# 🧠 Nexus: Intelligent Expense Estimator

### 🏆 *Winner - Second Prize at Hack the Future 2025, IIT Gandhinagar*

An AI-powered system for accurate and instant estimation of **Monthly Per Capita Expenditure (MPCE)** and **Total Household Expense** using socio-economic survey data.

---

## 📌 Problem Statement

Given household and individual-level data from the **HCES (Household Consumer Expenditure Survey)** dataset, develop a solution to:

* Predict total household expenses and MPCE
* Provide regional analytics by state, district, and NSS region
* Enable intuitive natural language querying for expense-related insights

---

## 🚀 Our Solution

Team **Nexus** built a fully functional **Flask API** that:

* Delivers **sub-second** CPU-only predictions using stacked **XGBoost** models
* Uses **custom composite indices** generated from grouped binary possessions/purchases
* Differentiates **normal** vs **top-5% outlier** households using classification + separate regression pipelines
* Supports **natural language** queries through a chatbot powered by **Groq (LLaMA-3.2)**
* Provides **region-wise analysis** (state, district, NSS) for decision-making and policy insights

---

## 🔍 Key Features

* 📊 **MPCE & Total Expense Prediction** (R² = 0.70, MAPE ≈ 23%)
* 📈 **Top 10 Feature Contributions** included in report
* 🔍 **Household Lookup** by ID
* 🌐 **Regional Distribution Analysis** (State, District, NSS Region)
* 🤖 **Chatbot Interface** for non-technical users to ask natural-language questions
* ⚡ **CPU-Only Deployment** (no GPU dependency)
* 🐳 **Containerizable** with Docker for easy deployment

---

## 🛠️ Tech Stack

* **Python, Flask** (API Layer)
* **XGBoost, RandomForest, CatBoost** (Modeling)
* **Pandas, NumPy** (Data Processing)
* **Scikit-learn** (Preprocessing, Pipelines)
* **Groq API** (LLM Chatbot)
* **Jupyter Notebooks** (Exploratory Data Analysis & Regression)
* **Joblib, Pickle** (Model Serialization)

---

## 📁 Project Structure

```
├── app.py                       # Main Flask API
├── chat.py                      # Chatbot logic (LLM + RAG)
├── classification_train.py      # Classifier for top-5% households
├── regression_train.ipynb       # MPCE regression model training
├── feature_engineering_notebook.ipynb
├── feature_retrieval_arc.py     # Composite feature extraction models
├── data/                        # HCES dataset (HH + person-level)
├── model/                       # Pretrained XGBoost/Classifier models
├── Presentation.pptx            # Project overview slides
├── Nexus_Flowchart2.drawio.png  # Pipeline diagram
```

---

## 📦 Setup Instructions

```bash
# Clone the repo
git clone https://github.com/your-username/nexus-expense-estimator.git
cd nexus-expense-estimator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the API
python app.py
```

---

## 🧪 API Endpoints

| Endpoint              | Description                           |
| --------------------- | ------------------------------------- |
| `/predict`            | Predicts MPCE and Total Expense       |
| `/search`             | Get household details by ID           |
| `/analyze_state`      | Analyze expense distribution by state |
| `/analyze_district`   | Analyze by district                   |
| `/analyze_nss_region` | Analyze by NSS region                 |
| `/chat`               | Ask natural language questions (Groq) |

---

## 🏅 Awards & Achievements

🏆 **Second Prize Winner** – [Hack the Future 2025](https://iitgn.ac.in), IIT Gandhinagar
🗓️ *March 21–23, 2025*
👨‍💻 **Team Nexus**:

* Sayuj Gupta (CSE) – Team Leader, Feature Engineering and Model Testing
* Himanshu (CSE) – Web Development and Feature Engineering
* D Barghav (Mech) – Dataset Preparation and Model Training
* Purushartha Gupta (Civil) – Feature Engineering and Model Training
* Parth Sachdeva (CSE) – Feature Engineering and Model Training

![Alt Text](photo.jpg)

---

## 📈 Future Scope

* 🗂️ Add multilingual support for chatbot queries
* 🧠 Integrate deeper NLP capabilities for trend explanations
* 🛰️ Expand data coverage with satellite-derived poverty metrics
* 📱 Mobile-friendly frontend for rural outreach

---
