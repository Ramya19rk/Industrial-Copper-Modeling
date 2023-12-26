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


