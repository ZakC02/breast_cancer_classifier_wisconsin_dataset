import pandas as pd
import streamlit as st
from utils.dataloader import load_data
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@st.cache_data
def preprocess_data(df):
    '''
    Preprocess the breast cancer dataset by dropping unnecessary columns and encoding the target variable.
    returns the features (X) and target (y) as separate variables.
    '''
    df.drop(columns=["id", "Unnamed: 32"], inplace=True)
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
    le = LabelEncoder()
    y = le.fit_transform(y)
    return X, y

# split the data into features and target
@st.cache_data
def split_data(X, y):
    '''
    Split the features and target into training and testing sets.
    returns the training and testing sets as separate variables.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

# normalize the features
@st.cache_data
def normalize_data(X_train, X_test):
    '''
    Normalize the features using StandardScaler.
    returns the normalized training and testing sets as separate variables.
    '''
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

@st.cache_data
def prepare_data():
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train, X_test = normalize_data(X_train, X_test)

    return X_train, X_test, y_train, y_test
