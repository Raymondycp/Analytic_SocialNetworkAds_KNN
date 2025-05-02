# Social Network Ads Classification

In the realm of digital marketing, understanding which users are likely to engage with advertisements is crucial for optimizing campaign performance. This project predicts whether users on a social network will purchase a product based on their age and estimated salary. By analyzing user demographics and applying supervised machine learning, the notebook provides insights into customer behavior, enabling targeted advertising strategies.

## Overview

This project focuses on building a classification model to predict whether a user will purchase a product advertised on a social network. The K-Nearest Neighbors (KNN) algorithm is used to classify users based on their `Age` and `EstimatedSalary`, with the target variable being `Purchased` (0 or 1). 

## Objectives

- **Exploratory Data Analysis (EDA)**: Investigate the distribution and relationships of `Age`, `EstimatedSalary`, `Gender`, and `Purchased` to understand user characteristics.  
- **Model Development**: Implement and train a KNN classifier to predict purchase behavior based on `Age` and `EstimatedSalary`.  
- **Visualization**: Create visualizations to explore data patterns and model performance.  
- **Business Insights**: Provide recommendations for targeting high-potential users in advertising campaigns.

## Dataset

The dataset, `Social_Network_Ads.csv`, contains 400 user records with the following columns:

- `User ID`: Unique identifier for each user (15566689 to 15815236).  
- `Gender`: User gender (Male or Female).  
- `Age`: User age (18 to 60 years).  
- `EstimatedSalary`: Estimated annual salary in dollars (15,000 to 150,000).  
- `Purchased`: Binary target variable indicating whether the user purchased the product (0 \= No, 1 \= Yes).

**Key Details**:

- The dataset includes categorical (`Gender`, `Purchased`) and numerical (`Age`, `EstimatedSalary`) features.  
- No missing values are present, as inferred from the clean dataset output.  
- The analysis focuses on `Age` and `EstimatedSalary` for model training, with `Gender` used in EDA but not necessarily in the KNN model.  
- The dataset is balanced for classification, with approximately 35.75% of users having purchased the product (mean of `Purchased` \= 0.3575).

## Key Steps

1. **Data Loading and Preprocessing**:  
     
   - The dataset is loaded using `pandas` into a DataFrame (`ad_data`).  
   - EDA includes:  
     - Descriptive statistics via `ad_data.describe()` to summarize `Age`, `EstimatedSalary`, and `Purchased`.  
     - Visualizations (inferred as typical for this dataset) such as histograms for `Age` and `EstimatedSalary`, and scatter plots of `Age` vs. `EstimatedSalary` colored by `Purchased` or `Gender`.

   

2. **Model Training**:  
     
   - **Feature Selection**: `Age` and `EstimatedSalary` are used as features for the KNN classifier (inferred as standard for this dataset).  
   - **Data Preprocessing**: Features are scaled using `StandardScaler` from `scikit-learn` to ensure KNN performs effectively (inferred as a common step).  
   - **KNN Classifier**: The KNN model is trained with an optimal `k` value (e.g., determined via cross-validation or elbow method, typically `k=5` for this dataset).  
   - **Train-Test Split**: The dataset is split into training and testing sets (e.g., 80-20 split) to evaluate model performance.

   

3. **Evaluation**:  
     
   - Model performance is assessed using metrics such as accuracy, precision, recall, and F1-score (inferred as standard for classification tasks).  
   - A confusion matrix and classification report are generated to analyze true positives, false positives, etc.  
   - Decision boundaries are visualized (inferred as typical for KNN with two features) to show how the model separates `Purchased` and `Not Purchased` classes.

---

