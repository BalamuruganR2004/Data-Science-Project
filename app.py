
import streamlit as st
import joblib
import numpy as np

model = joblib.load('titanic_model.pkl')

st.title("🚢 Titanic Survival Prediction")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouses Aboard", 0, 10, 0)
parch = st.number_input("Parents/Children Aboard", 0, 10, 0)
fare = st.number_input("Fare", 0.0, 500.0, 32.0)
embarked_q = st.selectbox("Embarked from Queenstown?", ["No", "Yes"])
embarked_s = st.selectbox("Embarked from Southampton?", ["No", "Yes"])

sex_val = 1 if sex == "Female" else 0
embarked_q_val = 1 if embarked_q == "Yes" else 0
embarked_s_val = 1 if embarked_s == "Yes" else 0

if st.button("Predict Survival"):
    features = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_q_val, embarked_s_val]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    if prediction == 1:
        st.success(f"✅ Survived! (Probability: {round(probability[1]*100, 2)}%)")
    else:
        st.error(f"❌ Did Not Survive (Probability: {round(probability[0]*100, 2)}%)")