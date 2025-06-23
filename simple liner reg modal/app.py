import streamlit as st
import pickle
import numpy as np

#load the saved model

model= pickle.load(open(r'D:\Naresh IT foundation\Python project\simple liner reg modal\Linear_regression_model.pkl', 'rb'))

#set the title of the streamlit app
st.title("Salary Prediction App")

# Add a brief description
st.write("This app predicts the salary based on years of experience using a linear regression model.")

#And input widget for user to enter years of experience
years_experience = st.number_input("Enter Years of Experience:", min_value=0.0, max_value=50.0,value=1.0,step=0.5)
 
 
#when the button is clicked, make a prediction
if st.button("Predict Salary"):
    #make a prediction using the model
    experience_input= np.array([[years_experience]])# coversion of input to 2D array for prediction
    prediction =model.predict(experience_input)
    
    
    #Display the result 
    st.success(f"The predicted salary for {years_experience} years of experience is: ${prediction[0]:.2f}")
    
#Display the results
st.write('the model is trained on a dataset of salaries based on years of experience. The model uses linear regression to make predictions.')   