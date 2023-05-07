import streamlit as st
from matplotlib import image
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pickle import load

scaler = load(open('models/standard_scaler.pkl', 'rb'))
lr_model = load(open('models/dt_model.pkl', 'rb'))

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
dir_of_interest = os.path.join(FILE_DIR, "resources")
IMAGE_PATH = os.path.join(dir_of_interest, "image")
DATA_PATH = os.path.join(dir_of_interest, "data", "penguins_lter.csv")
imgg= os.path.join(IMAGE_PATH, "images.jpg")

df = pd.read_csv(DATA_PATH)
st.title("Home")

img = image.imread(imgg)
st.image(img)

st.dataframe(df)
st.title("describe")
st.write(df.describe())

cl = st.text_input('Culmen Length (mm) ')
cd = st.text_input('Culmen Depth (mm) ')
fl = st.text_input('Flipper Length (mm) ')
bm = st.text_input('Body Mass (g) ')

btn_click = st.button("Predict")

if btn_click == True:
    if cl and cd and fl and bm:
        query_point = np.array([float(cl), float(cd), float(fl), float(bm)]).reshape(1, -1)
        query_point_transformed = scaler.transform(query_point)
        pred = lr_model.predict(query_point_transformed)
        st.success(pred)
    else:
        st.error("Enter the values properly.")