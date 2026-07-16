import streamlit as st
import pandas as pd
import plotly.express as px

from utils.dataloader import load_data

st.title("🩺 Breast Cancer Wisconsin Dataset")

st.set_page_config(
    page_title="Breast Cancer Wisconsin Dataset",
    page_icon="🩺",
)

df = load_data()
st.dataframe(df, width='stretch')



