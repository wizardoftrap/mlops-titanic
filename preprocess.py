import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.copy()
    
    #Drop rows with missing target variable
    if 'Survived' in df.columns:
        df = df.dropna(subset=['Survived'])
    
    #Fill missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    #Drop columns not needed for modeling
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, errors='ignore')
    
    #Encode categorical variables
    categorical_cols = ['Sex', 'Embarked']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
    
    return df
