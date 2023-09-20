{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "4f33c64e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pickle\n",
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "6d965572",
   "metadata": {},
   "outputs": [],
   "source": [
    "#loading Model\n",
    "load_model = pickle.load(open('Tree_model.sav','rb'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "36a5b52e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2023-09-19 19:29:59.428 Session state does not function when running a script without `streamlit run`\n"
     ]
    }
   ],
   "source": [
    "\n",
    "def heart_disease_prediction(input_data):\n",
    "    \n",
    "\n",
    "    #changing input data to numpy array\n",
    "    input_data_as_array = np.asarray(input_data)\n",
    "\n",
    "    #reshape array\n",
    "    input_data_reshape = input_data_as_array.reshape(1,-1)\n",
    "\n",
    "    prediction = load_model.predict(input_data_reshape)\n",
    "    print(prediction)\n",
    "\n",
    "    if (prediction[0] == 0):\n",
    "        return'The person is does not have heart diisease'\n",
    "    else:\n",
    "        return 'The person have heart diisease'\n",
    "    \n",
    "    \n",
    "def main():\n",
    "    #giving title\n",
    "    st.title('Heart Disease Prediction')\n",
    "    \n",
    "    #getting names of variable\n",
    "    #Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,HeartDisease\n",
    "    \n",
    "    # Input fields\n",
    "    Age = st.slider('Age', 20, 100)   # age range between 20 to 100\n",
    "    Sex = st.radio('Sex', ['Male', 'Female'])   #give select option from choice\n",
    "    Chest_Pain_Type = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-Anginal Pain', 'Asymptomatic'])\n",
    "    RestingBP = st.slider('Resting Blood Pressure', 80, 200)\n",
    "    Cholesterol = st.slider('Cholesterol', 100, 600)\n",
    "    FastingBS = st.radio('Fasting Blood Sugar > 120 mg/dl', ['No', 'Yes'])\n",
    "    RestingECG = st.selectbox('Resting ECG', ['Normal', 'ST-T Wave Abnormality', 'Left Ventricular Hypertrophy'])\n",
    "    MaxHR = st.slider('Max Heart Rate', 60, 220)\n",
    "    ExerciseAngina = st.radio('Exercise-Induced Angina', ['No', 'Yes'])\n",
    "    Oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.0)\n",
    "    ST_Slope = st.selectbox('ST Slope', ['Upsloping', 'Flat', 'Downsloping'])\n",
    "    \n",
    "    # Convert categorical input to numeric\n",
    "    Sex = 1 if Sex == 'Male' else 0\n",
    "    chest_pain_mapping = {'Typical Angina': 3, 'Atypical Angina': 1, 'Non-Anginal Pain': 2, 'Asymptomatic': 0}\n",
    "    resting_ecg_mapping = {'Normal': 1, 'ST-T Wave Abnormality': 2, 'Left Ventricular Hypertrophy': 0}\n",
    "    FastingBS = 1 if FastingBS == 'Yes' else 0\n",
    "    ExerciseAngina = 1 if ExerciseAngina == 'Yes' else 0\n",
    "    st_slope_mapping = {'Upsloping': 2, 'Flat': 1, 'Downsloping': 0}\n",
    "\n",
    "    # Predict heart disease\n",
    "    input_data = [Age, Sex, chest_pain_mapping[Chest_Pain_Type], RestingBP, Cholesterol, FastingBS,\n",
    "                  resting_ecg_mapping[RestingECG], MaxHR, ExerciseAngina, Oldpeak, st_slope_mapping[ST_Slope]]\n",
    "    \n",
    "    #code for prediction\n",
    "    result = ''\n",
    "    \n",
    "    #creating button for prediction\n",
    "    if st.button('Disease Test Result'):\n",
    "        result = heart_disease_prediction(input_data)\n",
    "        \n",
    "        st.success(result)\n",
    "        \n",
    "        \n",
    "if __name__ == \"__main__\":\n",
    "    main()\n",
    "\n",
    "\n",
    "    \n",
    "    \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "824557ed",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "392da648",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50066417",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
