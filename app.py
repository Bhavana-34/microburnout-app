import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load the trained ML model (replace with your model file)
model = joblib.load('microburnout_model.pkl')

# Set page configuration for better UI/UX
st.set_page_config(page_title="MicroBurnout Prediction", page_icon=":guardsman:", layout="wide")

# Add custom CSS for styling
st.markdown("""
    <style>
        /* General styling */
        body {
            background-color: #f0f4f7;
            font-family: 'Arial', sans-serif;
        }

        .title {
            color: #4CAF50;
            font-size: 36px;
            font-weight: bold;
        }

        .header {
            color: #2d3e50;
            font-size: 28px;
        }

        .subtitle {
            color: #00796b;
            font-size: 22px;
        }

        .input-section {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .prediction-section {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .sidebar .sidebar-content {
            background-color: #ffffff;
            padding: 30px;
            border-radius: 10px;
        }

        .btn-predict {
            background-color: #4CAF50;
            color: white;
            font-size: 18px;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px;
            width: 100%;
            cursor: pointer;
        }

        .btn-predict:hover {
            background-color: #45a049;
        }

    </style>
""", unsafe_allow_html=True)

# Set the title of the web app
st.title("MicroBurnout Prediction App :guardsman:")

# Add a brief description
st.write("""
    This web app uses machine learning to predict the likelihood of **MicroBurnout** based on personal input data.
    Fill in the details below and click on **Predict** to see the results!
""")

# Sidebar: Collect user input
st.sidebar.header('Enter Your Data:')
age = st.sidebar.number_input('Age', min_value=18, max_value=100, value=25, step=1)
hours_worked = st.sidebar.number_input('Hours Worked', min_value=1, max_value=24, value=8, step=1)
stress_level = st.sidebar.slider('Stress Level (0 = Low, 10 = High)', 0, 10, 5)
sleep_hours = st.sidebar.number_input('Hours of Sleep', min_value=0, max_value=24, value=7, step=1)

# Display the input data in a formatted way
st.markdown('---')
st.subheader("### Your Input Data")
st.write(f"**Age:** {age} years")
st.write(f"**Hours Worked per Day:** {hours_worked} hours")
st.write(f"**Stress Level:** {stress_level} (0 to 10 scale)")
st.write(f"**Hours of Sleep per Day:** {sleep_hours} hours")
st.markdown('---')

# Prepare the input data as a feature vector for the model
input_data = np.array([age, hours_worked, stress_level, sleep_hours]).reshape(1, -1)

# Prediction button
if st.sidebar.button('Predict', key='predict_button'):

    # Show a loading spinner while predicting
    with st.spinner('Predicting... please wait.'):

        # Make prediction using the model
        prediction = model.predict(input_data)

        # Display the result in the main section
        st.markdown('---')
        st.subheader("### Prediction Result")
        
        if prediction[0] == 1:
            st.write("""
                **Result:** You are likely experiencing microburnout. 
                It's crucial to take care of yourself. Please consider taking breaks, improving your work-life balance, 
                and focusing on mental well-being. :relieved:
            """)
        else:
            st.write("""
                **Result:** You are not likely experiencing microburnout. Keep up the great work! :muscle:
            """)
        
        st.markdown('---')

# Footer with credits and extra info
st.markdown("""
    <div style="text-align: center; padding-top: 20px;">
        <small>Created with ❤️ . All rights reserved.</small>
    </div>
""", unsafe_allow_html=True)
