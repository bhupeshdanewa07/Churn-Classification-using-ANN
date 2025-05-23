import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------ 🎨 App UI Personalization -------------------
st.set_page_config(page_title="Churn Prediction by Bhupesh", page_icon="📊")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>A Streamlit app developed by <span style='color: #2C7BE5;'>Bhupesh</span></h4>", unsafe_allow_html=True)

with st.sidebar:
    st.header("👨‍💻 About Bhupesh")
    st.markdown("""
    - 🎓 M.Tech in AI @ NIT Bhopal  
    - 💡 Focused on building intelligent automation solutions  
    - 📫 Connect: [LinkedIn](https://linkedin.com/in/bhupesh-danewa-17a65a204)  
    """)
# ---------------------------------------------------------------

# User inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance', min_value=0.0, format="%.2f")
credit_score = st.number_input('Credit Score', min_value=0)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format="%.2f")
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Predict button
if st.button('Predict'):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

    # Combine with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = prediction[0][0]

    st.success(f'🔍 Churn Probability: {prediction_proba:.2f}')

    if prediction_proba > 0.5:
        st.error('🚨 The customer is likely to churn.')
    else:
        st.info('✅ The customer is not likely to churn.')
else:
    st.write('ℹ️ Please enter details and click Predict.')

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center;'>Made with ❤️ by Bhupesh</div>", unsafe_allow_html=True)
