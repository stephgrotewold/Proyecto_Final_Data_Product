from flask import Flask, request, jsonify
import pandas as pd
import logging
import joblib
import getpass

app = Flask(__name__)

# Cargar el modelo
modelo = joblib.load('rus_random_forest.pkl')

# Configurar el registro
logging.basicConfig(filename='app.log', level=logging.INFO)

@app.route('/')
def base():
    return 'Prediction Api - Credit Risk Analysis'

# Endpoint para predecir
@app.route('/predict', methods=['POST'])
def predict_loan_approval():
    data = request.get_json()

    input_data = pd.DataFrame({
        'FLAG_OWN_CAR': [data['FLAG_OWN_CAR']],
        'FLAG_OWN_REALTY': [data['FLAG_OWN_REALTY']],
        'CNT_CHILDREN': [data['CNT_CHILDREN']],
        'AMT_INCOME_TOTAL': [data['AMT_INCOME_TOTAL']],
        'NAME_INCOME_TYPE': [data['NAME_INCOME_TYPE']],
        'NAME_EDUCATION_TYPE': [data['NAME_EDUCATION_TYPE']],
        'NAME_FAMILY_STATUS': [data['NAME_FAMILY_STATUS']],
        'NAME_HOUSING_TYPE': [data['NAME_HOUSING_TYPE']],
        'DAYS_BIRTH': [data['DAYS_BIRTH']],
        'DAYS_EMPLOYED': [data['DAYS_EMPLOYED']],
        'OCCUPATION_TYPE': [data['OCCUPATION_TYPE']],
        'CNT_FAM_MEMBERS': [data['CNT_FAM_MEMBERS']],
        'CNT_ADULTS': [data['CNT_ADULTS']],
        'AMT_INCOME_PER_CHILDREN': [data['AMT_INCOME_PER_CHILDREN']],
        'AMT_INCOME_PER_FAM_MEMBER': [data['AMT_INCOME_PER_FAM_MEMBER']]
    })

    # Realizar la predicción
    prediction = modelo.predict(input_data)

    # Mapear la predicción a "bueno" o "malo"
    if prediction[0] == 0:
        result = "bueno"
    else:
        result = "malo"

    # Preparar la respuesta
    response = {'prediction': result}

# Obtener el usuario y la dirección IP
    username = getpass.getuser()
    ip_address = request.remote_addr

    # Generar el registro
    logging.info({
        'usuario': username,
        'direccion_ip': ip_address,
        'end_point': request.path,
        'user_agent': request.user_agent.string,
        'time': str(pd.Timestamp.now()),
        'payload': data,
        'output': response
    })


    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)