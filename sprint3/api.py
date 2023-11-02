from flask import Flask, request, jsonify
import pandas as pd
import logging
import joblib
import getpass
import os
import datetime
import json

app = Flask(__name__)

# Cargar el modelo
modelo = joblib.load('rus_random_forest.pkl')

# Obtener la ruta actual
ruta_actual = os.path.dirname(os.path.abspath(__file__))

# Crear el directorio de registro si no existe
log_directory = os.path.join(ruta_actual, 'logs')
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configurar el registro fuera de la función de registro
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Crear un controlador de archivo por cada punto final
def create_file_handler(endpoint):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    log_file = os.path.join(log_directory, f"log_{endpoint}_{timestamp}.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    return file_handler

@app.route('/')
def base():
    return 'Prediction Api - Credit Risk Analysis'

# Endpoint para predecir un solo registro
@app.route('/predict', methods=['POST'])
def predict_loan_approval():
    data = request.get_json()
    endpoint = 'predict'

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

    prediction = modelo.predict(input_data)
    result = "bueno" if prediction[0] == 0 else "malo"
    response = {'prediction': result}

    file_handler = create_file_handler(endpoint)

    logger.info({
        'usuario': getpass.getuser(),
        'direccion_ip': request.remote_addr,
        'end_point': request.path,
        'user_agent': request.user_agent.string,
        'time': str(pd.Timestamp.now()),
        'payload': json.dumps(data),
        'output': response
    })

    logger.removeHandler(file_handler)

    return jsonify(response)

# Endpoint para predecir un lote de registros
@app.route('/predict_batch', methods=['POST'])
def predict_loan_approval_batch():
    data = request.get_json()
    endpoint = 'predict_batch'
    predictions = []

    for item in data:
        input_data = pd.DataFrame({
            'FLAG_OWN_CAR': [item['FLAG_OWN_CAR']],
            'FLAG_OWN_REALTY': [item['FLAG_OWN_REALTY']],
            'CNT_CHILDREN': [item['CNT_CHILDREN']],
            'AMT_INCOME_TOTAL': [item['AMT_INCOME_TOTAL']],
            'NAME_INCOME_TYPE': [item['NAME_INCOME_TYPE']],
            'NAME_EDUCATION_TYPE': [item['NAME_EDUCATION_TYPE']],
            'NAME_FAMILY_STATUS': [item['NAME_FAMILY_STATUS']],
            'NAME_HOUSING_TYPE': [item['NAME_HOUSING_TYPE']],
            'DAYS_BIRTH': [item['DAYS_BIRTH']],
            'DAYS_EMPLOYED': [item['DAYS_EMPLOYED']],
            'OCCUPATION_TYPE': [item['OCCUPATION_TYPE']],
            'CNT_FAM_MEMBERS': [item['CNT_FAM_MEMBERS']],
            'CNT_ADULTS': [item['CNT_ADULTS']],
            'AMT_INCOME_PER_CHILDREN': [item['AMT_INCOME_PER_CHILDREN']],
            'AMT_INCOME_PER_FAM_MEMBER': [item['AMT_INCOME_PER_FAM_MEMBER']]
        })

        prediction = modelo.predict(input_data)
        result = "bueno" if prediction[0] == 0 else "malo"
        predictions.append({'input': item, 'prediction': result})

        file_handler = create_file_handler(endpoint)

        logger.info({
            'usuario': getpass.getuser(),
            'direccion_ip': request.remote_addr,
            'end_point': request.path,
            'user_agent': request.user_agent.string,
            'time': str(pd.Timestamp.now()),
            'payload': json.dumps(item),
            'output': {'input': item, 'prediction': result}
        })

        logger.removeHandler(file_handler)

    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
