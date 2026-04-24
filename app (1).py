import streamlit as st
import pandas as pd
import joblib

# 1. Page Styling
st.set_page_config(page_title="AI-Bridge Sales Predictor", layout="centered")

st.title("📈 Advertising Sales Predictor")
st.write("Enter the budget details to predict total sales units.")

try:
    # 2. Load the trained model
    model = joblib.load('advertising_model.pkl')

    # 3. Create a Layout with Columns for User Input
    col1, col2, col3 = st.columns(3)

    with col1:
        tv = st.number_input("TV Budget ($)", min_value=0.0, max_value=500.0, value=0.0)

    with col2:
        radio = st.number_input("Radio Budget ($)", min_value=0.0, max_value=100.0, value=37.8)

    with col3:
        newspaper = st.number_input("Newspaper Budget ($)", min_value=0.0, max_value=200.0, value=69.2)

    # 4. Create a 'Predict' button
    if st.button("Calculate Prediction"):
        # Create a DataFrame from the dynamic user input
        user_input = pd.DataFrame([{
            'TV': tv,
            'radio': radio,
            'newspaper': newspaper
        }])

        # Get prediction
        prediction = model.predict(user_input)

        # 5. Display Result in a nice box
        st.divider()
        st.subheader("Results")
        st.metric(label="Estimated Sales Units", value=f"{prediction[0]:.2f}")

        # Show how the input compares
        st.bar_chart(user_input.T)

except Exception as e:
    st.error(f"Model Error: {e}")
