import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model

st.title("Bienvenido a la Calculadora de contaminantes del aire por costo de servicio de Taxi para la ciudad de Nueva York")
st.markdown('El objetivo de esta calculadora es implementar y proveer una herramienta para GREYHOUND en la que puede calcular las cantidades aproximadas de contaminantes del aire que se generarán dependiendo del valor de los servicios de sus Taxis.')



promediosAire = pd.read_csv("promediosaire.csv")
montos_diarios = pd.read_csv("montos_diarios.csv")

precio = st.slider('Precio por servicio en USD', 0, 80, 19)

#Modelo de regresión con Monoxido de Carbono
regresion = linear_model.LinearRegression()
precio_promedio = montos_diarios['total_amount_avg'].values.reshape((-1,1))
modelo = regresion.fit(precio_promedio,promediosAire['Carbon Monoxide'])
predice_CO = [[precio]]
modelo.predict(predice_CO)
if st.checkbox('mostrar Monóxido de Carbono'):
    st.write(modelo.predict(predice_CO))

#Modelo de regresión con Dioxido de Nitrogeno
regresion = linear_model.LinearRegression()
precio_promedio = montos_diarios['total_amount_avg'].values.reshape((-1,1))
modelo = regresion.fit(precio_promedio,promediosAire['Nitrogen dioxide'])
predice_NO2 = [[precio]]
modelo.predict(predice_NO2)
if st.checkbox('mostrar Dióxido de Nitrógeno'):
    st.write(modelo.predict(predice_NO2))


#Modelo de regresión con Dioxido de Azufre
regresion = linear_model.LinearRegression()
precio_promedio = montos_diarios['total_amount_avg'].values.reshape((-1,1))
modelo = regresion.fit(precio_promedio,promediosAire['Sulfur dioxide'])
predice_SO2 = [[precio]]
modelo.predict(predice_SO2)
if st.checkbox('mostrar Dióxido de Azufre'):
    st.write(modelo.predict(predice_SO2))