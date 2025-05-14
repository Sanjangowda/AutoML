import os

import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from ml_utility import (
    read_data,
    preprocess_data,
    train_model,
    evaluate_model
)

# Set Streamlit page config - must be the first Streamlit command
st.set_page_config(
    page_title="AutoML",
    page_icon="🧠",
    layout="centered")

# Custom CSS for Glassmorphism UI
st.markdown("""
    <style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 2rem;
        margin: 1rem 0;
        color: #ffffff;
    }

    .glass-button {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 0.5rem 1rem;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .glass-button:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Get working and parent directory
working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)


st.title("🧠 Automated ML Model Training Platform")

# Dataset selection
dataset_list = os.listdir(f"{parent_dir}/data")
dataset = st.selectbox("Select a dataset from the dropdown", dataset_list, index=None)



if dataset:
    df = read_data(dataset)

    if df is not None:
        st.dataframe(df.head())

        col1, col2, col3, col4 = st.columns(4)

        scaler_type_list = ["standard", "minmax"]

        model_dictionary = {
            "Logistic Regression": LogisticRegression(),
            "Support Vector Classifier": SVC(),
            "Random Forest Classifier": RandomForestClassifier(),
            "XGBoost Classifier": XGBClassifier()
        }

        with col1:
            target_column = st.selectbox("Select the Target Column", list(df.columns))
        with col2:
            scaler_type = st.selectbox("Select a scaler", scaler_type_list)
        with col3:
            selected_model = st.selectbox("Select a Model", list(model_dictionary.keys()))
        with col4:
            model_name = st.text_input("Model name")

        st.markdown('<div class="glass-button">', unsafe_allow_html=True)

        if st.button("Train the Model"):
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column, scaler_type)

            model_to_be_trained = model_dictionary[selected_model]

            model = train_model(X_train, y_train, model_to_be_trained, model_name)

            accuracy = evaluate_model(model, X_test, y_test)

            st.success("Test Accuracy: " + str(accuracy))

            # Download button for trained model
            with open(f"{parent_dir}/trained_model/{model_name}.pkl", "rb") as f:
                st.download_button(
                    label="Download Trained Model",
                    data=f,
                    file_name=f"{model_name}.pkl",
                    mime="application/octet-stream"

                )

        st.markdown('</div>', unsafe_allow_html=True)
