# Industrial-Copper-Modeling

Problem Statement: Develop and deploy a machine learning model for predicting manufacturing-related outcomes, incorporating data preprocessing, advanced regression and classification techniques, feature engineering, and creating an interactive web application using Streamlit to facilitate real-time predictions and decision-making in the manufacturing domain.

NAME : RAMYA KRISHNAN A

BATCH: DW75DW76

DOMAIN : DATA SCIENCE

DEMO VIDEO URL : https://www.linkedin.com/posts/ramyakrishnan19_excited-to-unveil-my-latest-project-activity-7145280722537185280-gu2y?utm_source=share&utm_medium=member_desktop

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

<img width="1440" alt="Screenshot 2023-12-26 at 10 24 01 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/19289a7e-ac9a-4e1a-b366-64fb9b3c0863">


# Selling Price Menu:

Form to input parameters like Quantity Ton, Customer Type, Country Code, etc.

Upon submitting the form, the model predicts and displays the Selling Price.

<img width="1440" alt="Screenshot 2023-12-26 at 10 24 16 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/87428beb-6519-48e3-88c1-9f4c5aab0dea">

<img width="1440" alt="Screenshot 2023-12-26 at 10 24 25 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/329f1f2a-1e1a-4553-abd8-83ec110aee11">

<img width="1440" alt="Screenshot 2023-12-26 at 10 25 06 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/9548f8cd-8fa5-4a98-a849-782be027330b">

<img width="1440" alt="Screenshot 2023-12-26 at 10 25 16 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/5e463d6b-00b1-4dfb-b39d-63e4939648e7">


# Status Menu:

Form to input parameters like Quantity Ton, Customer Type, Country Code, etc.

Upon submitting the form, the model predicts and displays the Status (e.g., Won or Lost).

<img width="1440" alt="Screenshot 2023-12-26 at 10 25 31 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/41238164-1d24-4022-a3d8-a73472e06204">

<img width="1440" alt="Screenshot 2023-12-26 at 10 25 38 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/dc21abd2-046c-4df3-8a3a-4bcd788d4f7f">

<img width="1439" alt="Screenshot 2023-12-26 at 10 25 46 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/a4518d67-d034-448d-bd43-8ef1fd780b90">

<img width="1440" alt="Screenshot 2023-12-26 at 10 25 53 AM" src="https://github.com/Ramya19rk/Industrial-Copper-Modeling/assets/145639838/1fbd192f-f2e2-453d-88cf-45f8e25873ee">


# Learning Outcomes

Proficiency in Python programming and data analysis libraries.

Experience in data preprocessing techniques, EDA, and machine learning modeling.

Building interactive web applications using Streamlit.

