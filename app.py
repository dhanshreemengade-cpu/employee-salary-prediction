import streamlit as st
import joblib
import numpy as np

st.title("Salary Prediction App")

st.sidebar.header("Employee Details")


st.write("With this app, you can get estimations for the salaries of the company employees")

years = st.number_input("Years of Experience", value=1, step=1, min_value=0)

jobrate = st.number_input("Job Rate", value=3.5, step=0.5, min_value=0.0)

X = [years, jobrate]

model = joblib.load("Linearmodel.pkl")

st.divider()

predict = st.button("Press the button for salary prediction")

st.divider()

if predict:
    st.balloons()

    X1 = np.array([X])

    prediction = model.predict(X1)[0]

    st.write(f"Salary prediction is {prediction:,.2f}")

else:
    st.write("Please press the button for app to make the prediction")


