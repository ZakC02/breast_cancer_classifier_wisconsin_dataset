import streamlit as st
import pandas as pd
import plotly.express as px

from utils.dataloader import load_data

st.title("Exploratory Data Analysis")

df = load_data()

counts = (
    df["diagnosis"]
    .value_counts()
    .rename_axis("Diagnosis")
    .reset_index(name="Count")
)

fig = px.bar(
    counts,
    x="Diagnosis",
    y="Count",
    text="Count",
    title="Target Variable Distribution",
    color = 'Diagnosis',
    color_discrete_map={
        "B": "#1f77b4",  # Blue
        "M": "#d62728",  # Red
    },
    )

fig.update_traces(textposition="outside")
fig.update_layout(
    showlegend=False,
    xaxis_title="Diagnosis",
    yaxis_title="Number of Samples",
    height=450,
)

st.plotly_chart(fig)

st.subheader("Distribution Comparison")

feature = st.selectbox("Feature Selection", df.columns[2:], key="feature_select")

if feature:
    fig2 = px.histogram(
        df,
        x=feature,
        color="diagnosis",
        barmode="overlay",
        color_discrete_map={
        "B": "#1f77b4",  # Blue
        "M": "#d62728",  # Red
    },
        title=f"Distribution of {feature} by Diagnosis",
    )
    fig2.update_layout(
        xaxis_title=feature,
        yaxis_title="Count",
        height=450,
    )
    st.plotly_chart(fig2)