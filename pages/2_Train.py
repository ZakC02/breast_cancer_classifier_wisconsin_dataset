import streamlit as st
from utils.dataloader import load_data
from utils.preprocess import preprocess_data, split_data, normalize_data, prepare_data
from utils.train import train_lr, train_xgb, save_model

st.title("Train a Model")

@st.cache_resource
def cached_train_lr(X_train, y_train):
    return train_lr(X_train, y_train)


@st.cache_resource
def cached_train_xgb(X_train, y_train):
    return train_xgb(X_train, y_train)



if "loaded" not in st.session_state:
    st.session_state.loaded = False

with st.spinner('loading data...'):
    X_train, X_test, y_train, y_test = prepare_data()
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.loaded = True
    st.success('Data loaded successfully!')

model_type = st.selectbox("Select Model Type", ["Logistic Regression", "XGBoost"])

if st.button("Train"):
    if not st.session_state.loaded:
        st.error("Please load the data first.")
    else:
        with st.spinner(f'training {model_type} model ... '):
            if model_type == "Logistic Regression":
                model = cached_train_lr(X_train, y_train)
            elif model_type == "XGBoost":
                model = cached_train_xgb(X_train, y_train)

        save_model(model, model_type)
        st.success('Model trained and saved successfully!')

