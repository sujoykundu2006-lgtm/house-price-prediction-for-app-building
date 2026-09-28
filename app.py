"""
Module 5: AI-Powered Real Estate Estimator & Smart Explainer
Interactive Streamlit Application using Scikit-Learn
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI House Price Predictor", layout="wide", page_icon="🏠")

@st.cache_resource
def load_and_train_model():
    """Load dataset, preprocess, and train the Random Forest model."""
    df = pd.read_csv("House Price Prediction Dataset.csv")
    if "Id" in df.columns:
        df.drop(columns=["Id"], inplace=True)
        
    df_processed = pd.get_dummies(df, columns=["Location", "Condition", "Garage"], drop_first=True)
    
    X = df_processed.drop(columns=["Price"])
    y = df_processed["Price"]
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X.columns.tolist()

st.title("🏠 AI-Powered House Price Predictor & Insights")
st.markdown(
    "This web app demonstrates machine learning applied to housing valuation, "
    "developed with assistance from modern AI development tools."
)

try:
    model, feature_names = load_and_train_model()
    data_loaded = True
except Exception as e:
    st.error(f"Please ensure 'House Price Prediction Dataset.csv' is present in the repository. Error: {e}")
    data_loaded = False

if data_loaded:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Property Specifications")
        area = st.slider("Total Area (sq ft)", min_value=500, max_value=6000, value=2500, step=50)
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
        floors = st.selectbox("Number of Floors", [1, 2, 3], index=1)
        year_built = st.slider("Year Built", min_value=1900, max_value=2024, value=2010)
        
        col_a, col_b = st.columns(2)
        with col_a:
            location = st.selectbox("Location Type", ["Downtown", "Rural", "Suburban", "Urban"])
            garage = st.radio("Has Garage?", ["Yes", "No"])
        with col_b:
            condition = st.selectbox("Property Condition", ["Excellent", "Good", "Fair", "Poor"])

    with col2:
        st.subheader("💡 Valuation & AI Insights")
        
        # Build property dataframe
        input_dict = {feat: 0 for feat in feature_names}
        input_dict["Area"] = area
        input_dict["Bedrooms"] = bedrooms
        input_dict["Bathrooms"] = bathrooms
        input_dict["Floors"] = floors
        input_dict["YearBuilt"] = year_built
        
        # Set dummy variables
        if f"Location_{location}" in input_dict:
            input_dict[f"Location_{location}"] = 1
        if f"Condition_{condition}" in input_dict:
            input_dict[f"Condition_{condition}"] = 1
        if garage == "Yes" and "Garage_Yes" in input_dict:
            input_dict["Garage_Yes"] = 1
            
        input_df = pd.DataFrame([input_dict])
        
        predicted_price = model.predict(input_df)[0]
        
        st.metric(
            label="Estimated Market Value", 
            value=f"${predicted_price:,.2f}"
        )
        
        st.divider()
        st.markdown("### 🤖 Automated Property Summary")
        summary_text = (
            f"A **{condition.lower()}** condition property located in a **{location.lower()}** area "
            f"with **{area:,} sq ft**, featuring **{bedrooms} bedrooms** and **{bathrooms} bathrooms**.\n\n"
            f"- **Valuation Assessment:** Priced at approximately **${predicted_price/area:,.2f}/sq ft**.\n"
            f"- **Age Factor:** Constructed in **{year_built}** ({2026 - year_built} years old).\n"
            f"- **Garage Amenity:** {'Included' if garage == 'Yes' else 'Not included'}."
        )
        st.info(summary_text)