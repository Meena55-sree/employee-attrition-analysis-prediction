# Employee Attrition Analysis & Prediction

A machine learning web application that analyzes employee attrition patterns and predicts whether an employee is likely to leave an organization.

##  Project Overview

Employee attrition is an important challenge for organizations because unexpected employee turnover can increase recruitment costs and affect productivity.

This project uses **Machine Learning and Data Analysis** to:

* Analyze employee attrition patterns
* Identify important factors influencing employee turnover
* Predict employee attrition risk
* Evaluate the performance of the machine learning model
* Provide HR insights and recommendations

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Streamlit
* Joblib

##  Machine Learning Model

The project uses a **Random Forest Classifier** to predict employee attrition.

### Model Evaluation

* Accuracy: approximately **83%**
* ROC-AUC: approximately **0.79**

The exact values may vary slightly depending on the model execution and dataset split.

##  Dashboard Features

###  Overview

Provides an overview of the employee dataset and important HR statistics.

###  Attrition Prediction

Allows users to enter employee information and predict attrition risk.

###  Model Analysis

Displays:

* Model accuracy
* ROC-AUC score
* Model performance information

###  HR Insights
Provides:

* Important factors influencing attrition
* Dataset-level insights
* Attrition statistics
* HR recommendations

##  Dataset

The project uses an employee attrition dataset containing information about employees such as:

* Age
* Monthly Income
* Job Role
* Department
* Business Travel
* Overtime
* Job Satisfaction
* Distance From Home
* Years at Company
* Total Working Years
* And other employee-related attributes

Dataset size: **1470 employees**

## Project Structure

```text
employee-attrition-analysis-prediction/
│
├── app.py
├── HR_Employee_Attrition.csv
├── attrition_model.pkl
├── requirements.txt
├── README.md
│
├── Employee Attrition-prediction.png
├── dashboard.png
├── prediction-result.png
└── confusion-matrix.png
```

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd employee-attrition-analysis-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

##  Live Application

The project is deployed using Streamlit.

**Live Demo:** <your-streamlit-app-url>

##  Project Objective

The main objective of this project is to demonstrate how machine learning can be applied to HR analytics to identify employee attrition patterns and support data-driven decision making.

##  Developed By

**Meena Sree V T**

B.Tech Information Technology
St. Joseph's Institute of Technology
