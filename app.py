
import streamlit as st
import pandas as pd
import joblib

# Load saved model
saved = joblib.load("decision_tree_model.pkl")
model = saved["model"]
features = saved["features"]

st.title("Insurance Cost Prediction")
st.write("Decision Tree Machine Learning Model")

# User inputs
age = st.number_input("Age", min_value=1, max_value=120, value=25)
sex = st.selectbox("Sex", ["male", "female"])
bmi = st.number_input("BMI", min_value=1.0, value=25.0)
children = st.number_input("Children", min_value=0, max_value=20)
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox(
    "Region",
    ["southwest", "southeast", "northwest", "northeast"]
)

if st.button("Predict Insurance Cost"):
    data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": region
    }])

    data = pd.get_dummies(data)
    data = data.reindex(columns=features, fill_value=0)

    prediction = model.predict(data)

    st.success(f"Predicted Insurance Cost: ${prediction[0]:,.2f}")
