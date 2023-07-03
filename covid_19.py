import streamlit as st
import sklearn
import pickle
import pandas as pd
import numpy as np
from PIL import Image

covid_19_prediction = pickle.load(open('linear_model_for covid19.sav', 'rb'))

st.title('Covid 19 active cases prediction app')

st.sidebar.header('Covid19 Data')

def user_report():
  infected = st.sidebar.slider('infected', 1,300000, 1)
  tested = st.sidebar.slider('tested', 1,10000000, 1 )
  recovered= st.sidebar.slider('recovered', 1,300000, 1 )
  deceased= st.sidebar.slider('deceased', 1,5000, 1 )

  user_report_data = {
      'infected': infected,
      'tested':tested,
      'recovered':recovered,
      'deceased': deceased,
  }
  report_data = pd.DataFrame(user_report_data, index=[0])
  return report_data

user_data = user_report()
st.header('Covid19 Data')
st.write(user_data)

activeCases = covid_19_prediction.predict(user_data)
st.subheader(np.round(activeCases[0], 2))

