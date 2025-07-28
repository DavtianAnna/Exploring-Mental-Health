# 🧠 Mental Health Prediction - Kaggle Classification Challenge (Playground Series S4E11)

This repository contains my solution for the [Playground Series - S4E11](https://www.kaggle.com/competitions/playground-series-s4e11/overview) Kaggle competition.  
The task is to predict the likelihood of an individual suffering from **Depression**, based on various personal, academic, and professional attributes.  
This is a **binary classification** problem, where the model aims to predict whether the target variable `Depression` is 0 (No) or 1 (Yes).

---

## 🗂️ Project Structure

՝՝՝

project-root/
├── 📄 mental_health.ipynb # Main notebook: full pipeline from preprocessing to model training & submission.
├── 📊 train.csv # Training dataset including target (Depression).
├── 🧪 test.csv # Test dataset for prediction.
├── 📝 sample_submission.csv # Kaggle submission format.
├── 🚀 submission.csv # Final output predictions for Kaggle.
└── 📜 README.md # Project documentation.


՝՝՝


---

## 💻 Technologies Used

- **Programming Language**: Python 3.x
- **Notebook Environment**: Jupyter Notebook
- **Libraries**:
  - `pandas`, `numpy`: Data manipulation
  - `seaborn`, `matplotlib`: Visualization (e.g. boxplots, distributions)
  - `scikit-learn`: Machine learning framework
    - `StandardScaler`: Feature scaling
    - `SVC` (Support Vector Classifier): Core ML model
    - `Pipeline`: To streamline preprocessing + model
    - `GridSearchCV`: Hyperparameter tuning
    - `LabelEncoder`: Encoding categorical features
    - `train_test_split`, `classification_report`, `ConfusionMatrixDisplay`: Model evaluation
  - `warnings`: For suppressing unnecessary logs

---

## 📊 Dataset Description

The dataset contains anonymized responses regarding individuals’ lifestyle, education, job satisfaction, and mental health status.

- `train.csv`: Labeled data including:
  - Features like `Gender`, `Sleep Duration`, `Dietary Habits`, `Financial Stress`, `Study Satisfaction`, etc.
  - Target: `Depression` (0 or 1)
- `test.csv`: Same features, without the target.
- `sample_submission.csv`: Required format for submission.

---

## 🔁 Project Workflow

The `mental_health.ipynb` notebook includes the following steps:

### 1. 📥 Data Loading & Cleaning
- Loaded the data from `.csv` files.
- Dropped non-informative columns like `Name` and `City`.
- Cleaned categorical values by ensuring test set categories are consistent with training set.
- Filled missing values:
  - Categorical: mode
  - Numeric: median

### 2. 🔡 Encoding
- Applied `LabelEncoder` to convert categorical variables to numeric form.
- Ensured label consistency between train and test sets.

### 3. 📊 EDA (Exploratory Data Analysis)
- Used boxplots and value counts to inspect data distribution.
- Checked for imbalance in target classes.

### 4. 🤖 Model Building - Support Vector Classifier (SVC)
- Used `Pipeline` with `StandardScaler` and `SVC`.
- Tuned hyperparameters with `GridSearchCV`:
  - `C`: `[1, 5, 10]`
  - `kernel`: `['linear', 'poly', 'rbf', 'sigmoid']`
  - `gamma`: `['scale', 'auto']`

### 5. 🧪 Evaluation
- Split data into training and validation sets (80/20).
- Evaluated model using:
  - `classification_report` (precision, recall, F1)
  - `ConfusionMatrixDisplay`

### 6. 📤 Prediction & Submission
- Final predictions generated on the test dataset.
- Used `.predict_proba()` if required for probability output or `.predict()` for binary class.
- Saved results in `submission.csv` for Kaggle upload.

---

## 📈 Performance Summary

| Model            | Algorithm         | Evaluation Metric | Cross-Validation | Output File       |
|------------------|-------------------|-------------------|------------------|-------------------|
| SVC (with scaling) | Support Vector Machine | Accuracy, F1-score | 3-Fold CV via GridSearch | `submission.csv` |

*Note: Accuracy depends heavily on data preprocessing and feature encoding consistency.*

---

## ⚙️ Installation

Install all required dependencies:
