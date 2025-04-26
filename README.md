# HR Analytics: Churn Prediction using Python and Power BI

## Overview
This project aims to predict employee churn (whether an employee will leave the company) using **Logistic Regression** in **Python** and visualize the actual vs predicted churn using **Power BI**.  
It provides HR departments with actionable insights to help reduce attrition and improve retention strategies.

---

## Project Workflow

1. **Data Collection**:
   - Used a dataset `HR_comma_sep.csv` containing employee information such as satisfaction level, salary, department, etc.
   
2. **Data Preprocessing**:
   - Encoded categorical variables (`Departments`, `salary`) using `LabelEncoder`.
   - Scaled feature values using `StandardScaler`.
   
3. **Model Building**:
   - Built a **Logistic Regression** model to predict employee attrition (`left` column).
   - Split data into training and testing sets.
   
4. **Prediction**:
   - Predicted whether employees are likely to leave.
   - Calculated prediction probabilities.

5. **Visualization**:
   - Imported results into **Power BI**.
   - Created visualizations to compare actual vs predicted churn.

---

## Python Code

```python
# Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler



# Encode categorical features
le = LabelEncoder()
dataset['Departments'] = le.fit_transform(dataset['Departments'])
dataset['salary'] = le.fit_transform(dataset['salary'])

# Select features and target
x = dataset[['satisfaction_level', 'last_evaluation', 'number_project',
             'average_montly_hours', 'time_spend_company', 'Work_accident',
             'promotion_last_5years', 'Departments', 'salary']]
y = dataset['left']

# Feature scaling
sc = StandardScaler()
x = sc.fit_transform(x)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log = LogisticRegression()
log.fit(X_train, y_train)

# Make predictions
y_pred = log.predict(x)
y_prob = log.predict_proba(x)[:,1]

# Add predictions back to the dataset
dataset['predictions'] = y_pred
dataset['probabilities'] = y_prob

# Save updated dataset for visualization
dataset.to_csv('HR_file_updated.csv', index=False)
