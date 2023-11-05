import streamlit as st
import requests
import pandas as pd
import json
import matplotlib.pyplot as plt

st.title("Credit Risk Analysis Dashboard")

# Crear selector de tabs
selected_tab = st.sidebar.radio("Selecciona una pestaña:", ["Telemetría del modelo", "Evaluación del modelo", "Batch Scoring Test", "Atomic Scoring Test"])

# Telemetría del modelo
if selected_tab == "Telemetría del modelo":
    st.header("Telemetría del modelo")

    # Número de veces que el modelo es llamado
    model_calls = 0
    st.subheader("Número de veces que el modelo es llamado")
    st.write(model_calls)

    # Número de filas predichas
    predicted_rows = 0
    st.subheader("Número de filas predichas")
    st.write(predicted_rows)

    # Tiempo promedio de respuesta
    average_response_time = 0
    st.subheader("Tiempo promedio de respuesta")
    st.write(average_response_time)

# Evaluación del modelo
elif selected_tab == "Evaluación del modelo":
    st.header("Evaluación del modelo")

    # Métricas y su historia
    st.subheader("Métricas y su historia")
    # Puedes agregar gráficos o tablas con las métricas y su historia aquí

# Batch Scoring Test
elif selected_tab == "Batch Scoring Test":
    st.header("Batch Scoring Test")

    # Carga de un archivo para predecir
    st.subheader("Carga de un archivo para predecir")
    file = st.file_uploader("Cargar archivo CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df)

        # Realiza la predicción desde el archivo cargado
        if st.button("Predict from File"):
            if "modelo" in globals():
                df = pd.read_csv(file)

                # Realiza la predicción utilizando el modelo cargado
                predictions = modelo.predict(df)  # Asegúrate de que la lógica de predicción sea la correcta

                # Crea un DataFrame con las predicciones
                result_df = pd.DataFrame({"Prediction": predictions})

                # Muestra las predicciones
                st.subheader("Predictions")
                st.dataframe(result_df)

            else:
                st.write("El modelo no está cargado. Carga un modelo antes de realizar la predicción.")

# Atomic Scoring Test
elif selected_tab == "Atomic Scoring Test":
    st.header("Atomic Scoring Test")

    # Predicción de un solo record
    st.subheader("Predicción de un solo record")

    # Crea Streamlit widgets para ingresar datos de predicción
    # Agrega los campos necesarios y el botón para realizar la predicción

  # Crea Streamlit widgets para ingresar datos de predicción
    # Flag Own Car
    flag_own_car = st.selectbox("Flag Own Car", ["Y", "N"])
    flag_own_realty = st.selectbox("Flag Own Realty", ["Y", "N"])
    cnt_children = st.number_input("Number of Children", min_value=0)
    amt_income_total = st.number_input("Total Income")
    name_income_type = st.selectbox("Name Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
    name_education_type = st.selectbox("Name Education Type", ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"])
    name_family_status = st.selectbox("Name Family Status", ["Civil marriage", "Married", "Single / not married", "Separated", "Widow"])
    name_housing_type = st.selectbox("Name Housing Type", ["Rented apartment", "House / apartment", "Municipal apartment", "With parents", "Co-op apartment", "Office apartment"])
    days_birth = st.number_input("Days Birth")
    days_employed = st.number_input("Days Employed")
    occupation_type = st.selectbox("Occupation Type", ["nan", "Security staff", "Sales staff", "Accountants", "Laborers", "Managers", "Drivers", "Core staff", "High skill tech staff", "Cleaning staff", "Private service staff", "Cooking staff", "Low-skill Laborers", "Medicine staff", "Secretaries", "Waiters/barmen staff", "HR staff", "Realty agents", "IT staff"])
    cnt_fam_members = st.number_input("Count Family Members")
    cnt_adults = st.number_input("Count Adults")
    amt_income_per_children = st.number_input("Income per Child")
    amt_income_per_fam_member = st.number_input("Income per Family Member")


    if st.button("Predict Single Record"):
        # Prepara los datos del registro de entrada
        input_data = {
            'FLAG_OWN_CAR': flag_own_car,
            'FLAG_OWN_REALTY': flag_own_realty,
            'CNT_CHILDREN': cnt_children,
            'AMT_INCOME_TOTAL': amt_income_total,
            'NAME_INCOME_TYPE': name_income_type,
            'NAME_EDUCATION_TYPE': name_education_type,
            'NAME_FAMILY_STATUS': name_family_status,
            'NAME_HOUSING_TYPE': name_housing_type,
            'DAYS_BIRTH': days_birth,
            'DAYS_EMPLOYED': days_employed,
            'OCCUPATION_TYPE': occupation_type,
            'CNT_FAM_MEMBERS': cnt_fam_members,
            'CNT_ADULTS': cnt_adults,
            'AMT_INCOME_PER_CHILDREN': amt_income_per_children,
            'AMT_INCOME_PER_FAM_MEMBER': amt_income_per_fam_member
        }

        # Realiza la predicción para el registro de entrada
        response = requests.post("http://localhost:5000/predict", json=input_data)  # Ajusta la URL según tu configuración

        # Muestra el resultado de la predicción
        if response.status_code == 200:
            prediction = response.json()['prediction']
            st.write(f"Prediction: {prediction}")
        else:
            st.write("Error al obtener una predicción")
