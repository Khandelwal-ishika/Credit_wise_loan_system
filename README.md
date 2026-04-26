# 💳 CreditWise Loan System

> An intelligent ML-powered loan approval system built for **SecureTrust Bank** — predicting loan approvals accurately, faster, and without human bias.

---

## 🏦 Problem Statement

**SecureTrust Bank** offers personal and home loans to customers across urban and rural regions of India. Every day, hundreds of customers apply for loans through online and branch applications.

Until now, the bank relied on a **manual verification process** where loan officers evaluate applications by checking income proofs, employment details, credit history, and other documents. This process is **time-consuming, biased, and inconsistent.**

This led to two major business challenges:
1. **Good customers sometimes get rejected** → loss of business
2. **High-risk customers sometimes get approved** → financial losses

**Solution:** An intelligent loan approval system powered by Machine Learning that automatically analyses applicant details and **predicts whether a loan should be Approved or Rejected** before final human verification.

---

## 📌 Objective

> Design and develop an ML system using historical loan application data that learns hidden patterns from previous customer records and provides **accurate, fast, and unbiased loan approval decisions.**

---

## 🗂️ Dataset Description

**File:** `loan_approval_data.csv`
**Size:** 1000 rows × 20 columns | 950 non-null entries

Each row represents a **loan applicant** with personal, financial, and credit information.

| Column | Description |
|---|---|
| Applicant_ID | Unique applicant ID |
| Applicant_Income | Monthly income of applicant |
| Coapplicant_Income | Monthly income of co-applicant |
| Employment_Status | Salaried / Self-Employed / Business |
| Age | Applicant age |
| Marital_Status | Married / Single |
| Dependents | Number of dependents |
| Credit_Score | Credit bureau score |
| Existing_Loans | Number of already running loans |
| DTI_Ratio | Debt-to-Income ratio |
| Savings | Savings balance |
| Collateral_Value | Value of collateral provided |
| Loan_Amount | Loan amount requested |
| Loan_Term | Loan duration (months) |
| Loan_Purpose | Home / Education / Personal / Business |
| Property_Area | Urban / Semi-Urban / Rural |
| Education_Level | Graduate / Postgraduate / Undergraduate |
| Gender | Male / Female |
| Employer_Category | Govt / Private / Self |
| **Loan_Approved** | **Target: 1 = Approved, 0 = Rejected** |

**Class Distribution:**
- ❌ Not Approved (No): **70.2%**
- ✅ Approved (Yes): **29.8%**

---

## ✨ Project Highlights

- ✅ Full Exploratory Data Analysis (pie charts, bar plots, histograms, box plots)
- ✅ Missing value imputation — mean for numerical, mode for categorical
- ✅ Label Encoding + One-Hot Encoding for all categorical features
- ✅ Correlation heatmap — **Credit_Score (0.45)** is the strongest predictor
- ✅ Feature Engineering — `DTI_Ratio_sq` and `Credit_Score_sq`
- ✅ Train-Test Split (80/20) + StandardScaler normalization
- ✅ Three ML models trained, evaluated, and compared
- ✅ Best model selected on Precision → **Naive Bayes**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Seaborn + Matplotlib | EDA and data visualization |
| Scikit-learn | Preprocessing, ML models, evaluation |
| Jupyter Notebook | Development and experimentation |

---

## 📂 Project Structure

```
credit-wise-loan-system/
│
├── credit_wise.ipynb          # Main Jupyter notebook (complete pipeline)
├── loan_approval_data.csv     # Dataset (1000 applicants, 20 features)
├── requirements.txt           # Python dependencies
└── README.md
```

---

## ⚙️ ML Pipeline

```
Data Loading → EDA → Missing Value Handling → Feature Encoding →
Correlation Analysis → Feature Engineering → Train-Test Split →
Feature Scaling → Model Training → Evaluation → Best Model Selection
```

### Step 1 — Data Loading
- Loaded `loan_approval_data.csv` (1000 rows × 20 columns)
- Inspected with `df.info()` and `df.describe()`

### Step 2 — Exploratory Data Analysis (EDA)
- **Class balance:** 70.2% rejected, 29.8% approved
- **Gender:** Male 621, Female 379
- **Education:** Graduate 722, Not Graduate 278
- **Income distribution:** histograms for Applicant & Coapplicant income
- **Box plots:** Income, Credit Score, DTI Ratio vs Loan_Approved
- **Correlation heatmap:** Credit_Score (0.45) strongest positive predictor; Loan_Amount (-0.13) strongest negative

### Step 3 — Handle Missing Values
- 50 missing values per column (950/1000 non-null)
- `SimpleImputer(strategy="mean")` → numerical columns
- `SimpleImputer(strategy="most_frequent")` → categorical columns

### Step 4 — Feature Encoding
- Dropped `Applicant_ID` (non-informative)
- `LabelEncoder` → `Education_Level`, `Loan_Approved`
- `OneHotEncoder(drop="first")` → `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, `Employer_Category`
- Final feature count: **27 columns**

### Step 5 — Feature Engineering
- `DTI_Ratio_sq = DTI_Ratio ** 2`
- `Credit_Score_sq = Credit_Score ** 2`

### Step 6 — Train-Test Split + Scaling
- 80/20 split, `random_state=42`
- `StandardScaler` on train and test sets

---

## 📊 Model Results

### Round 1 — Base Features

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 86.5% | 78.3% | 77.0% | 77.7% |
| KNN (k=5) | 76.0% | 62.7% | 52.5% | 57.1% |
| **Naive Bayes** ✅ | **86.5%** | **80.4%** | 73.8% | 76.9% |

### Round 2 — After Feature Engineering

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 87.5% | 79.0% | 80.3% | 79.7% |
| KNN (k=5) | 75.5% | 62.0% | 50.9% | 55.9% |
| **Naive Bayes** ✅ | **86.5%** | **78.3%** | 77.0% | 77.7% |

> 🏆 **Best Model → Naive Bayes** (highest Precision — minimizes false approvals, critical for a bank)

### Top Features by Correlation with Loan Approval

| Feature | Correlation |
|---|---|
| Credit_Score | +0.45 |
| Applicant_Income | +0.12 |
| Employer_Category_MNC | +0.07 |
| Loan_Term | -0.09 |
| Loan_Amount | -0.13 |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Khandelwal-ishika/credit-wise-loan-system.git
cd credit-wise-loan-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the notebook

```bash
jupyter notebook credit_wise.ipynb
```

---

## 📋 requirements.txt

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

---

## 🧠 Key Learnings

- **Credit Score is the #1 predictor** — correlation of 0.45 with loan approval
- **Naive Bayes beats complex models** on precision — simplicity wins when the cost of false approvals is high
- **Feature engineering** (squared terms) improved Logistic Regression from 86.5% → 87.5%
- **Class imbalance (70/30)** must be considered — optimizing precision over accuracy matters for banks
- **One-Hot Encoding** categorical features like Employment Status and Employer Category adds meaningful signal

---

## 🙋 Author

Ishika Kahndelwal
- GitHub: [@Khandelwal-ishika](https://github.com/Khandelwal-ishika)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/ishikakhandelwal)

---
