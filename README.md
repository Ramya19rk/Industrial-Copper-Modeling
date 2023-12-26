# Industrial-Copper-Modeling

Problem Statement: Develop and deploy a machine learning model for predicting manufacturing-related outcomes, incorporating data preprocessing, advanced regression and classification techniques, feature engineering, and creating an interactive web application using Streamlit to facilitate real-time predictions and decision-making in the manufacturing domain.

NAME : RAMYA KRISHNAN A

BATCH: DW75DW76

DOMAIN : DATA SCIENCE

DEMO VIDEO URL : 

Linked in URL : www.linkedin.com/in/ramyakrishnan19

# Library used to handle the data

    import pandas as pd
    import numpy as np
    import sklearn 
    import matplotlib.pyplot as plt


# Library used for Classification Model

    import sklearn
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn import metrics
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc , accuracy_score
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    import matplotlib.pyplot as plt
    import pickle
    import pandas as pd

Classification Model is used to predict the Status

# Library used for Regression Model

    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    import pickle

Regression Model is used to predict the Selling Price


# Overview

This project involves the development of a machine learning model for industrial copper modelling, focusing on predicting key outcomes such as Selling Price and Status. The project integrates various data analysis, preprocessing, and machine learning techniques to provide valuable insights into the manufacturing domain.

# Domain and Technology Used

# Domain

The project addresses challenges in the manufacturing domain, aiming to leverage machine learning to enhance decision-making processes related to industrial copper.

# Technology Stack

Programming Language: Python

Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Streamlit

Machine Learning Models: ExtraTreesRegressor, XGBRegressor, ExtraTreesClassifier, Logistic Regression

Web Application Framework: Streamlit

# Project Structure

# Home Menu:

Overview of the project, its objectives, and significance.

Details about the manufacturing domain and the role of machine learning in solving related challenges.

# Selling Price Menu:

Form to input parameters like Quantity Ton, Customer Type, Country Code, etc.

Upon submitting the form, the model predicts and displays the Selling Price.

# Status Menu:

Form to input parameters like Quantity Ton, Customer Type, Country Code, etc.

Upon submitting the form, the model predicts and displays the Status (e.g., Won or Lost).

# Usage

# Home:
Provides an overview of the project, domain, and technology used.

# Selling Price:

Fill out the form with relevant details.

Click "Process" to get the predicted Selling Price.

# Status:

Fill out the form with relevant details.

Click "Process" to get the predicted Status.


# Learning Outcomes

Proficiency in Python programming and data analysis libraries.

Experience in data preprocessing techniques, EDA, and machine learning modeling.

Building interactive web applications using Streamlit.

