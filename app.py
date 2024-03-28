import pandas as pd
import numpy as np
import pickle
import streamlit as st
import joblib
from joblib import load
import time


def set_background_color():
    # Define your custom CSS 
    custom_css = """
    <style>
        .stApp {
            background-color:#051d30;
        }
         
    </style>
    """
    # Inject CSS with markdown
    st.markdown(custom_css, unsafe_allow_html=True)

# Invoke the function to set background color
set_background_color()

model = joblib.load('mlp_tuned.joblib')
knn_imputer=load('knn_imputer.joblib')

#Page Title
st.header(':blue[Patient Risk identification for Cerebral Stroke]',divider='grey')

def preprocess_inputs(user_inputs):

    # return processed_inputs
    if user_inputs['bmi_imputed'] >47.44:
        st.markdown("<span style='color: red'>BMI level is high!</span>", unsafe_allow_html=True)
    bmi_imputed = (user_inputs['bmi_imputed'] - 0) / (1 - 0)

    if user_inputs['avg_glucose_level'] > 162.645:
        st.markdown("<span style='color: red'>Glucose level is high!</span>", unsafe_allow_html=True)

    avg_glucose_level = (user_inputs['avg_glucose_level'] - 0) / (1 - 0)
    age = (user_inputs['age'] - 0) / (1 - 0)

    # Convert binary categorical inputs to boolean
    hypertension = True if user_inputs['hypertension'] == 'Yes' else False
    heart_disease = True if user_inputs['heart_disease'] == 'Yes' else False
    ever_married_yes = True if user_inputs['ever_married'] == 'Yes' else False
    residence_type_urban = True if user_inputs['residence_type'] == 'Urban' else False

    # One-hot encoding for multi-category inputs, ensuring the order matches the training data
    gender_male = True if user_inputs['gender'] == 'Male' else False
    gender_other = True if user_inputs['gender'] == 'Other' else False

    work_type_never_worked = True if user_inputs['work_type'] == 'Never worked' else False
    work_type_private = True if user_inputs['work_type'] == 'Private' else False
    work_type_self_employed = True if user_inputs['work_type'] == 'Self-employed' else False
    work_type_children = True if user_inputs['work_type'] == 'children' else False

    # Handling smoking status with numerical encoding
    if user_inputs['smoking_status_imputed'] == 'smokes':
        smoking_status_imputed = 1
    elif user_inputs['smoking_status_imputed'] == 'never smoked':
        smoking_status_imputed = 0
    else:  # Handles "" (unknown) and "formerly smoked" as middle value
        smoking_status_imputed = 0.5

 # Collecting all the processed features into a dictionary
    processed_features = {
        'age': [age],
        'hypertension': [hypertension],
        'heart_disease': [heart_disease],
        'avg_glucose_level': [avg_glucose_level],
        'gender_Male': [gender_male],
        'gender_Other': [gender_other],
        'ever_married_Yes': [ever_married_yes],
        'work_type_Never_worked': [work_type_never_worked],
        'work_type_Private': [work_type_private],
        'work_type_Self-employed': [work_type_self_employed],
        'work_type_children': [work_type_children],
        'Residence_type_Urban': [residence_type_urban],
        'smoking_status_imputed': [smoking_status_imputed],
        'bmi_imputed': [bmi_imputed]
    }

  # Convert the dictionary to a DataFrame
    processed_df = pd.DataFrame.from_dict(processed_features)

    return processed_df



st.sidebar.header('Patient Record')
 # Numeric inputs
age = st.sidebar.number_input("Age:", min_value=0, step=1,format="%d")
bmi_imputed = st.sidebar.number_input("BMI:")
avg_glucose_level = st.sidebar.number_input("Average Glucose Level:")
hypertension = st.sidebar.radio("Hypertension",["Yes", "No"],index = None, horizontal = True)
heart_disease = st.sidebar.radio("Heart Disease",["Yes", "No"],index = None, horizontal = True)

# Categorical inputs
smoking_status_imputed = st.sidebar.selectbox( "Smoking Status:", options=["", "formerly smoked", "never smoked", "smokes"],index = None, placeholder = "Select smoking status" )
work_type = st.sidebar.selectbox("Work Type:",options=["", "Private", "Self-employed", "Govt_job", "children", "Never worked"],index = None, placeholder = "Select work type")
residence_type = st.sidebar.selectbox("Residence Type:",options=["Urban", "Rural"],index = None, placeholder = "Select residence type")
gender = st.sidebar.radio("Gender",["Male", "Female", "Other"],index = None, horizontal = True)
ever_married = st.sidebar.radio(" Ever Married",["Yes", "No"],index = None, horizontal = True)

def make_prediction(model, input_data):
    # Preprocess the input data
    processed_data = preprocess_inputs(input_data)
    
    # Make a prediction
    prediction = model.predict(processed_data)
    
    return prediction

# Placeholder for the prediction result
result_placeholder = st.empty()
if st.sidebar.button('Predict Risk'):
    title_placeholder = st.empty()
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)

    # Example of different processing phases
    phases = [('Processing...', 20), ('Preparing input data...', 50), ('Making prediction...', 30)]
    start = 0

    for phase, duration in phases:
        title_placeholder.markdown(f"<h3 style='color:white;'>{phase}</h3>", unsafe_allow_html=True)
        for i in range(start, start + duration):
            time.sleep(0.03)  # Simulate processing time
            progress_bar.progress(i + 1)
        start += duration

    # Simulate making a prediction
    input_data = {
        'age': age,
        'bmi_imputed': bmi_imputed,
        'avg_glucose_level': avg_glucose_level,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_status_imputed': smoking_status_imputed,
        'work_type': work_type,
        'residence_type': residence_type,
        'gender': gender,
        'ever_married': ever_married
    }
    prediction = make_prediction(model, input_data)
    
    alert_stroke_image_url = 'stroke.png'
    healthy_brain_image_url = 'healthy_brain.png'
    if prediction[0]:  
        brain_image_url = alert_stroke_image_url
        message = "🚨 Higher risk of stroke"
        description = "The prediction indicates a higher risk of stroke. Please consult with a healthcare provider for further guidance.⚠️🚑👩‍⚕️🧠"
    else:
        brain_image_url = healthy_brain_image_url
        message = "✅ Lower risk of stroke"
        description = "The prediction indicates a lower risk of stroke. Continue maintaining a healthy lifestyle.👍💪🥦🏃‍♂️"


    # Clear placeholders for the progress and title
    progress_placeholder.empty()
    title_placeholder.empty()

    
    col1, col2 = st.columns([2, 2])

    with col1:
        st.image(brain_image_url, width=250,output_format="auto")

    with col2:
        st.markdown(f"""
        <div style = 'margin-top : 70px;'>
                     </div>
        <h3 style='color:white;'>{message}</h3>
        <p style='color:white;'>{description}</p>
       
        """, unsafe_allow_html=True)




    # Displaying recommondation
    def generate_health_tips(age, bmi, avg_glucose_level, hypertension, heart_disease, smoking_status):
        tips = []

    # Age-specific advice
        if age > 50:
            tips.append("👩‍⚕️ Regular check-ups are crucial as stroke risk increases with age.")
        else:
            tips.append("👦 Stroke risk is low at young age. Maintaining good health statistics prevents stroke at old age. ")
    
      # BMI-specific advice
        if bmi_imputed >= 25:
            tips.append("⚖️ Maintaining a healthy weight through diet and exercise can reduce stroke risk.")
        else:
            tips.append("🏃‍♂️ Continue maintaining a healthy weight.")
    
     # Glucose level advice
        if avg_glucose_level >= 100:
            tips.append("🩸 Monitoring and managing blood sugar levels can help prevent strokes.")
    
    # Hypertension advice
        if hypertension == "Yes":
            tips.append("❤️ Managing high blood pressure is key to reducing stroke risk.")
    
    # Heart disease advice
        if heart_disease == "Yes":
            tips.append("💔 Heart health is closely linked to stroke risk. Follow your doctor's advice closely.")
    
    # Smoking status
        if smoking_status_imputed == "smokes":
            tips.append("🚭 Quitting smoking can significantly reduce your risk of stroke.")
        elif smoking_status_imputed == "formerly smoked":
            tips.append("🚫 Not smoking is beneficial for your vascular health.")
        else:  # Non-smokers or never smoked
            tips.append("💨 Avoiding tobacco smoke helps reduce stroke risk.")

        return tips


    tips = generate_health_tips(age, bmi_imputed, avg_glucose_level, hypertension, heart_disease, smoking_status_imputed)


    st.markdown(f"<h3 style='color:white; background-color: #033f57;padding: 10px; margin-bottom:10px '> Health Tips</h3>", unsafe_allow_html=True)
    for tip in tips:
        st.markdown(f"<p style='color:white; background-color: #043359;padding:10px'> {tip}</p>", unsafe_allow_html=True)

