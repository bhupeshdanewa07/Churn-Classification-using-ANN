import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# —————————— CACHING & SETUP ——————————
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = tf.keras.models.load_model('model.h5')
    with open('label_encoder_gender.pkl','rb')  as f: le = pickle.load(f)
    with open('onehot_encoder_geo.pkl','rb') as f: ohe = pickle.load(f)
    with open('scaler.pkl','rb')             as f: sc = pickle.load(f)
    return model, le, ohe, sc

model, le_gender, ohe_geo, scaler = load_artifacts()

st.set_page_config(
    page_title="🚀 Churn Predictor by Bhupesh",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# —————————— SIDEBAR ——————————
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/4/4f/Maulana_Azad_National_Institute_of_Technology_Logo.png", use_container_width=True)
    st.markdown("## 👨‍💻 About")
    st.markdown("""
    - **Bhupesh Danewa**  
    - M.Tech AI @ NIT Bhopal  
    - Building Intelligent Automation Solutions  
    - [LinkedIn](https://linkedin.com/in/bhupesh-danewa-17a65a204)
    """)
    st.markdown("---")
    st.markdown("## ⚙️ Settings")
    theme = st.selectbox("Theme", ["Light","Dark"])
    if theme=="Dark":
        st.markdown(
            """<style>
               .css-1d391kg {background-color: #0e1117;}
             </style>""",
            unsafe_allow_html=True
        )

# —————————— MAIN LAYOUT ——————————
st.title("📊 Customer Churn Prediction")
st.markdown(
    "<p style='text-align:center; color:gray;'>Use this interactive tool to estimate churn risk.</p>",
    unsafe_allow_html=True
)

# two-column layout for inputs
col1, col2 = st.columns(2)
with col1:
    geography      = st.selectbox("Geography",      ohe_geo.categories_[0])
    gender         = st.selectbox("Gender",         le_gender.classes_)
    age            = st.slider("Age", 18, 92, 30)
    credit_score   = st.number_input("Credit Score", min_value=300, max_value=850, value=600)

with col2:
    balance        = st.number_input("Balance (₹)",   min_value=0.0, format="%.2f", value=50000.0)
    tenure         = st.slider("Tenure (Years)",    0, 10, 3)
    products       = st.slider("Number of Products",1, 4, 1)
    estimated_salary = st.number_input("Est. Salary (₹)", min_value=0.0, format="%.2f", value=50000.0)

has_cc   = st.radio("Has Credit Card?", ["Yes","No"])
active   = st.radio("Active Member?",   ["Yes","No"])

if st.button("🔍 Predict Churn"):
    # prepare input
    X = pd.DataFrame({
        'CreditScore':[credit_score], 'Gender':[le_gender.transform([gender])[0]],
        'Age':[age], 'Tenure':[tenure], 'Balance':[balance],
        'NumOfProducts':[products],
        'HasCrCard':[1 if has_cc=="Yes" else 0],
        'IsActiveMember':[1 if active=="Yes" else 0],
        'EstimatedSalary':[estimated_salary]
    })
    geo_enc = ohe_geo.transform([[geography]]).toarray()
    geo_df  = pd.DataFrame(geo_enc, columns=ohe_geo.get_feature_names_out())
    X_full  = pd.concat([X, geo_df], axis=1)
    X_scaled= scaler.transform(X_full)

    # predict
    proba = model.predict(X_scaled)[0,0]
    st.metric("Churn Probability", f"{proba:.2%}", delta=None)

    # show verdict
    if proba > 0.5:
        st.error("🚨 High churn risk!")
        st.markdown(
        """
        <div style="
            padding: 1rem;
            border: 2px solid #e63946;
            background-color: #f8d7da;
            color: #721c24;
            font-weight: bold;
            text-align: center;
            animation: blink 1s step-end infinite;
        ">
            ⚠️ WARNING: High churn risk detected! ⚠️
        </div>
        <style>
        @keyframes blink {
            0%, 100% { opacity: 1; }
            50%      { opacity: 0.2; }
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    else:
        st.success("✅ Low churn risk.")
        st.balloons()


# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;'>Made with ❤️ by Bhupesh</div>", unsafe_allow_html=True)
