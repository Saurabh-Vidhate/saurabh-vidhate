# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 07:41:11 2023

@author: COMPUTER
"""

import numpy as np
import pickle
import streamlit as st


#loading the save model
loade_model = pickle.load((open('‪‪F:/Model App/tree_model.sav','rb')))


#create prediction function
def Heart_dises_prediction(input_data):
    
    
    #convert input array to np array
    input_data_as_np_aaray = np.asarray(input_data)

    #reshape for predicting
    input_data_reshaped = input_data_as_np_aaray.reshape(1,-1)

    prediction = loade_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return "The person is not heart disease patient"
    else:
      return "The person is not heart disease patient"
  
    
def main():
    
    
    #getting a title
    st.title('Haert Disease Prediction Web App')
    
    #getting the input values
    
     # Input fields
    Age = st.slider('Age', 20, 100)   # age range between 20 to 100
    Sex = st.radio('Sex', ['Male', 'Female'])   #give select option from choice
    Chest_Pain_Type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])
    RestingBP = st.slider('Resting Blood Pressure', 80, 200)
    Cholesterol = st.slider('Cholesterol', 100, 600)
    FastingBS = st.radio('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])
    RestingECG = st.selectbox('Resting ECG', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])
    MaxHR = st.slider('Max Heart Rate', 60, 220)
    ExerciseAngina = st.radio('Exercise-Induced Angina', ['No', 'Yes'])
    Oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0)
    ST_Slope = st.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])
    
    # Convert categorical input to numeric
    Sex = 1 if Sex == 'Male' else 0
    chest_pain_mapping = {'Typical Angina': 3, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 0}
    resting_ecg_mapping = {'Normal': 1, 'ST-T Wave Abnormality': 2, 'Left Ventricular Hypertrophy': 0}
    FastingBS = 1 if FastingBS == 'Yes' else 0
    ExerciseAngina = 1 if ExerciseAngina == 'Yes' else 0
    st_slope_mapping = {'Upsloping': 2, 'Flat': 1, 'Downsloping': 0}

    # Predict heart disease
    input_data = [Age, Sex, chest_pain_mapping[Chest_Pain_Type], RestingBP, Cholesterol, FastingBS,
                  resting_ecg_mapping[RestingECG], MaxHR, ExerciseAngina, Oldpeak, st_slope_mapping[ST_Slope]]
    
    
    #code for prediction
    diagnosis = ''
    
    #creating a button for prediction
    
    
    if st.button('Heart Disease Result'):
        diagnosis = Heart_dises_prediction(input_data)
        st.success(diagnosis)
        
        
if __name__ == '__main__':
    main()
        
    
    