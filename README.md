# Credit Approval KNN Classifier (R)

This project uses the UCI Credit Approval dataset to build a K-Nearest Neighbors (KNN) machine learning classifier in R.

## Dataset
- Source: UCI Machine Learning Repository
- 690 credit applications with 15 attributes (both numeric and categorical)
- Includes missing values, anonymized features, and a binary approval status

## Techniques Used
- Data cleaning and preprocessing
- Handling missing values using the `mice` package (Predictive Mean Matching)
- Feature normalization for numeric columns
- Splitting into training and test sets
- Hyperparameter tuning (loop over K = 1 to 21)
- Model evaluation using test accuracy and confusion matrix
- Visualization of test error across K values

## Tools & Libraries
- R
- `mice`
- `class` (for KNN)
- Base plotting functions

## Results
- Optimal K value selected using test error plot
- Final model accuracy printed
- Confusion matrix displayed to show performance

## Files
- `CreditApproval_KNN.R`: Main R script with full data workflow
- `crx.data`: Original dataset (optional — can be downloaded directly if not included)

## What I Learned
- How to prepare real-world data for classification tasks
- The importance of handling missing values carefully
- How to evaluate a simple model with limited features
