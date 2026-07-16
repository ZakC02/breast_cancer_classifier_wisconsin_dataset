## prediction page

import streamlit as st
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

from utils.predict import predict_score

st.title("Model Score and Confusion Matrix")

if st.session_state.X_train is None or st.session_state.X_test is None or st.session_state.y_train is None or st.session_state.y_test is None:
    st.error("Please load the data first.")

# finds all the models in the models directory and lists them in a selectbox
import os

model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
model_names = [os.path.splitext(f)[0] for f in model_files]

selected_model = st.selectbox("Select a Model", model_names)

st.button("Predict", on_click=lambda: st.session_state.update({"selected_model": selected_model}))

if "selected_model" in st.session_state:
    model_file = f"models/{st.session_state.selected_model}.pkl"
    import joblib

    model = joblib.load(model_file)

    # make predictions on the test set
    accuracy, cm = predict_score(model, st.session_state.X_test, st.session_state.y_test)

    # display the predictions
    st.subheader("Predictions")
    st.write(f"Accuracy: {accuracy}")

    # display the confusion matrix with plotly
    st.subheader("Confusion Matrix")
    fig = ff.create_annotated_heatmap(
        z=cm,
        x=["Predicted Negative", "Predicted Positive"],
        y=["Actual Negative", "Actual Positive"],
        colorscale="Blues",
    )
    st.plotly_chart(fig)

