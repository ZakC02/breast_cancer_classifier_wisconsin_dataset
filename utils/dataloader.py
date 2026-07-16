import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    df = pd.read_csv("hf://datasets/scikit-learn/breast-cancer-wisconsin/breast_cancer.csv")
    return df