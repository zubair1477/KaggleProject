### Metastatic Breast Cancer Diagnosis Prediction

### Challenge
Predict whether a patient's metastatic breast cancer diagnosis period is less than 90 days by using the features in the given dataset. The ultimate goal is to have a robust model capable of identifying patterns in patient characteristics that influence timely diagnosis.

#### **Data Type:**
- **Input**: A CSV file containing patient information, medical history, geographic, socioeconomic, and environmental data. 
- **Output**: Binary target variable `DiagPeriodL90D` indicating whether a patient’s cancer diagnosis period is less than 90 days.
- **Size**: (12,906,83). ~10,000 after data cleaning
  - Training: ~8,000 samples.
  - Validation: ~2,500 samples.
  - Test: ~1,500 samples.

---

#### **Preprocessing / Cleanup**
- **Handling Missing Values**:
  - Numerical columns: Imputed using the median.
  - Categorical columns: Missing values filled proportionally based on other column distributions 
- **Encoding Categorical Variables**:
  - Label Encoding as there were already many columns
- **Feature Selection**:
  - Removed highly correlated geographic features to reduce redundancy.
  - Dropped irrelevant features that had no importance.
- **Class Imbalance**:
  - Initial dataset showed imbalance (~62% positive class). Applied weighted loss functions and SMOTE for balancing.

---

#### **Data Visualization**
- **Class Distribution**:
  - Highlighted imbalance between the two classes.
- **Feature Distributions**:
  - Visualized distributions of age, environmental factors (Ozone, PM25), and medical history across classes.
- **Correlation Heatmap**:
  - Identified relationships between important features like `diagnosis codes` and `socioeconomic factors`.
![image](https://github.com/user-attachments/assets/940f61b6-745c-468b-85d9-b957b7461033)
![image](https://github.com/user-attachments/assets/1164ab14-513c-4dbb-8fb1-d720a6965b74)
![image](https://github.com/user-attachments/assets/430a485e-dd67-47cc-9c2c-c07e3f65134f)


---

### **Problem Formulation**

- **Input/Output**:
  - Input: Preprocessed patient data.
  - Output: Binary classification (diagnosis within 90 days or not).
- **Model**: XGBoost Classifier for its robustness and best performance from 3 baseline models.
- **Evaluation Metrics**:
  - Primary: AUC-ROC.
  - Additional: Precision, Recall, F1-Score, PR Curve.
- **Baseline Models**:
  - Random Forest and Logistic Regression were compared with XGBoost for performance evaluation.

---

### **Training**

- **Process**:
  - Training split: 80% for training, 20% for validation.
  - Used GridSearchCV to optimize hyperparameters like `max_depth`, `learning_rate`, and `subsample`.
- **Duration**:
  - Training completed in ~15 minutes (including GridSearch).
- **Challenges**:
  - Balancing class distribution and ensuring feature consistency between train and test datasets.

---

### **Performance Comparison**

- **Metrics**:
  - AUC-ROC, precision, recall, confusion matrix, and feature importance.
- **Results**:
  - XGBoost outperformed baseline models:
    precision    recall  f1-score   support

           0       0.71      0.61      0.65       937
           1       0.79      0.85      0.82      1614

    accuracy                           0.76      2551
   macro avg       0.75      0.73      0.74      2551
weighted avg       0.76      0.76      0.76      2551

---

### **Conclusions**

- The XGBoost model showed some important features.
- Feature importance revealed key factors like `breast_cancer_diagnosis_code`, `age`, and `payer_type`.
- Class imbalance remains a challenge, but weighted loss functions improved recall for the minority class.
![image](https://github.com/user-attachments/assets/5efb1956-8ef8-4847-9cf2-c4fe1806f917)

---

### **Future Work**

1. **Improve Feature Engineering**:
   - Explore interaction terms between socioeconomic and environmental factors.
2. **Refine Model**:
3. **Enhance Class Balancing**:
   - experiment with undersampling techniques/other ways
4. **Real World Impact**:
    - Help the medical field analyze what factors into late/early diagnoses

---

### **How to Reproduce Results**

1. **Setup Instructions**:
   - Install required libraries
   - Download the dataset from the competition page.
2. **Training**:
   - Preprocess data: Handle missing values, encode categorical features
   - Run 
3. **Evaluation**:
   - Use the model to predict on the test set and generate a submission file.

---

### **Overview of Files in Repository**



---

### **Software Setup**

- **Required Packages**: pandas, scikit-learn, xgboost, matplotlib, seaborn.
- **Data Access**:
  - Dataset available on the competition platform.
  - https://www.kaggle.com/competitions/widsdatathon2024-challenge1/data

---

