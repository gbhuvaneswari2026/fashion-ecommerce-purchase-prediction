# Fashion E-Commerce Purchase Prediction
**Student ID:** A00105680 | **Module:** Data Analytics | **Coursework 2**

## Overview
This project applies machine learning to predict customer purchase behaviour 
using the H&M Personalised Fashion Recommendations dataset from Kaggle. 
Four classifiers are compared: Logistic Regression, Decision Tree, 
Random Forest, and Gradient Boosting.

## Dataset
Source: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

Files used:
- customers.csv (1,371,980 customer records)
- articles.csv (105,542 product records)

Note: transactions_train.csv (>500 MB) is not included. Behavioural 
features were simulated from a probabilistic model calibrated to the 
H&M dataset distributions (see Section 6.6 of the report).

## Files
- A00105680_Coursework2.py — main script (data prep, modelling, figures)
- articles.csv — product metadata
- customers.csv — customer demographics
- requirements.txt — Python dependencies

## How to Run
1. Install dependencies:
   pip install scikit-learn pandas matplotlib numpy

2. Place articles.csv and customers.csv in the project folder

3. Run:
   python A00105680_Coursework2.py

4. Figures are saved as PNG files in the working directory.

## Results Summary
| Model               | Accuracy | F1 Score | AUC   |
|---------------------|----------|----------|-------|
| Logistic Regression | 66.86%   | 76.91%   | 68.30%|
| Decision Tree       | 65.56%   | 76.33%   | 65.31%|
| Random Forest       | 66.34%   | 77.20%   | 67.60%|
| Gradient Boosting   | 66.74%   | 77.02%   | 68.04%|
