import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split as split
import sklearn.metrics as metrics
from imblearn.under_sampling import RandomUnderSampler

application_record = pd.read_csv('application_record.csv')
credit_record = pd.read_csv('credit_record.csv')

application_record.info()

application_record.nunique()
#Application
# Revisar valores duplicados
duplicates_bool = application_record.duplicated(subset='ID', keep=False)
print('There are',sum(duplicates_bool),'duplicates in ID column.')
duplicates = application_record[duplicates_bool].sort_values('ID')
duplicates.head(10)
duplicates['ID'].value_counts().max()
application_record_unique = application_record.drop_duplicates(subset='ID', keep=False, inplace=False)
application_record_unique.info()

#Credit
credit_record.info()
credit_record.nunique()
credit_decision = pd.DataFrame()
credit_decision['ID'] = credit_record['ID'].unique()

def get_credit_decision(id):
    #Determina, para cierto cliente con una ID, si el sliente es bueno o malo en base a su historial de pagos
    
    # Defiinir las constantes
    GOOD_STATUS = ['0']
    BAD_STATUS = ['1', '2', '3', '4', '5']
    UNUSED_STATUS = ['X']
    
    # Filtrar el record de creditos para el ID
    client_status = credit_record[credit_record['ID'] == id]['STATUS'].tolist()
    
    # Revisar si hay status malos
    if any(status in BAD_STATUS for status in client_status):
        return 1
    
    # Revisar si hay estatus no usados
    elif all(status in UNUSED_STATUS for status in client_status):
        return -1
    
    # Sino, asumir que es bueno
    else:
        return 0
    

# Crea una columna de decision en credit_decision dataset y llama a get_credit_decision() para obtener el cluster
credit_decision['Decision'] = credit_decision['ID'].map(get_credit_decision)
credit_decision.info()
credit_decision['Decision'].value_counts()

# Excluir a los clientes sin informacion suficiente (category -1) de credit_decision
credit_decision = credit_decision[credit_decision['Decision']!=-1]
credit_decision['Decision'].value_counts()

#Join de los data frames
join_data = pd.merge(credit_decision,application_record_unique)
join_data.head()
join_data.info()

columns = join_data.columns.difference(['ID'])
join_data['INFO'] = join_data[columns].apply(lambda x: '_'.join(x.map(str)), axis=1)
join_data['INFO'].nunique()
join_data['INFO'].value_counts().sort_values(ascending=False).head(1)
join_data[join_data['INFO']=='297000.0_0_1.0_F_-15519_-3234_0_0_1_N_Y_0_0_Secondary / secondary special_Single / not married_Rented apartment_Commercial associate_Laborers']

unique_join_data = join_data.drop_duplicates(subset='INFO', keep='first')
unique_join_data.drop('INFO', axis=1, inplace=True)
unique_join_data.info()
unique_join_data.nunique()

#Null Data
unique_join_data.isnull().sum()
unique_join_data[['DAYS_EMPLOYED']][unique_join_data['OCCUPATION_TYPE'].isnull()].value_counts()
def transform_days(value):
    return -value if value < 0 else 0
unique_join_data['DAYS_EMPLOYED'] = unique_join_data['DAYS_EMPLOYED'].apply(transform_days)
unique_join_data.loc[unique_join_data['DAYS_EMPLOYED'] == 0, 'OCCUPATION_TYPE'] = 'not_working'
unique_join_data.isnull().sum()
non_null_data = unique_join_data.fillna('Unknown')
non_null_data.info()
non_null_data['CNT_ADULTS'] = non_null_data['CNT_FAM_MEMBERS'] - non_null_data['CNT_CHILDREN']
non_null_data['CNT_ADULTS'].value_counts()
inconsistent_family_entries = non_null_data[non_null_data['CNT_ADULTS'] <= 0]
inconsistent_family_entries
non_null_data = non_null_data.drop(inconsistent_family_entries.index)
non_null_data['CNT_ADULTS'].value_counts()
non_null_data['DAYS_BIRTH'] = -non_null_data['DAYS_BIRTH']
non_null_data['DAYS_BIRTH']
credit_data = non_null_data.drop(['ID','CODE_GENDER', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL'], axis=1)
credit_data.info()
credit_data['AMT_INCOME_PER_CHILDREN'] = np.where(credit_data['CNT_CHILDREN'] > 0, credit_data['AMT_INCOME_TOTAL']/credit_data['CNT_CHILDREN'],0)
credit_data['AMT_INCOME_PER_CHILDREN'].describe()
credit_data['AMT_INCOME_PER_FAM_MEMBER'] = credit_data['AMT_INCOME_TOTAL']/credit_data['CNT_FAM_MEMBERS']
credit_data['AMT_INCOME_PER_FAM_MEMBER'].describe()
credit_data.info()

#  Separar variables categoricas
categorical_columns = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
categorical_data = credit_data[categorical_columns]
# Factorizar variables categoricas y transformarlas en data numerica
numerical_categorical_data = categorical_data.apply(lambda x: pd.factorize(x)[0])
numerical_categorical_data = pd.DataFrame(numerical_categorical_data)
numerical_categorical_data.head()
# Updatear los valores a credit_data data frame
credit_data[categorical_columns] = numerical_categorical_data
credit_data.head()

#Revisarl el data imbalance
credit_decision['Decision'].value_counts(normalize=True)
fig, ax = plt.subplots()
client = ['Good Clients (Decision = 0)','Bad Clients (Decision = 1)']
proportions = credit_decision['Decision'].value_counts(normalize=True)
bar_colors = ['tab:green','tab:red']
ax.bar(client, proportions, color=bar_colors)
ax.set_ylabel('Percentage of Dataset')
ax.set_title('Data Imbalance')
ax.text(1-0.1, proportions[1]/2, '{:.2%}'.format(proportions[1]), size=10)
ax.text(0-0.1, proportions[0]/2, '{:.2%}'.format(proportions[0]), size=10)
fig.show()

#Random Forest
y = credit_data['Decision']
X = credit_data.drop(['Decision'], axis=1)
unbal_X_train, X_test, unbal_y_train, y_test = split(X, y, test_size=0.2, random_state=0)
rus = RandomUnderSampler(sampling_strategy=(3/7), random_state=0)
rus_X_train, rus_y_train = rus.fit_resample(unbal_X_train, unbal_y_train)
fig, ax = plt.subplots()
client = ['Bad Clients','Good Clients']
proportions = rus_y_train.value_counts(normalize=True)
bar_colors = ['tab:green', 'tab:red']
ax.bar(client, proportions, color=bar_colors)
ax.set_ylabel('Percentage of Dataset')
ax.set_title('Resampled Data')
ax.text(1-0.1, proportions[1]/2, '{:.2%}'.format(proportions[1]), size=10)
ax.text(0-0.1, proportions[0]/2, '{:.2%}'.format(proportions[0]), size=10)
fig.show()

unbal_rf = RFC( n_estimators = 1000, max_features = 8, random_state=0)
unbal_rf.fit(unbal_X_train, unbal_y_train)

rus_rf = RFC( n_estimators = 1000, max_features = 8, random_state=0)
rus_rf.fit(rus_X_train, rus_y_train)



# Definir un caso de ejemplo (asegúrate de que los valores coincidan con el formato de tus datos)
# Ejemplo:
sample_data = pd.DataFrame({
    'FLAG_OWN_CAR': [0],
    'FLAG_OWN_REALTY': [1],
    'CNT_CHILDREN': [2],
    'AMT_INCOME_TOTAL': [50000],
    'NAME_INCOME_TYPE': [0],
    'NAME_EDUCATION_TYPE': [1],
    'NAME_FAMILY_STATUS': [2],
    'NAME_HOUSING_TYPE':[1],
    'DAYS_BIRTH': [12000],
    'DAYS_EMPLOYED': [3000],
    'OCCUPATION_TYPE': [3],
    'CNT_FAM_MEMBERS': [2],
    'CNT_ADULTS': [2],
    'AMT_INCOME_PER_CHILDREN': [103500],
    'AMT_INCOME_PER_FAM_MEMBER': [90000]
})

# Realizar predicción para el caso de ejemplo
prediction = rus_rf.predict(sample_data)

# Imprimir la predicción
print("Predicción para el caso de ejemplo:", prediction)

import joblib

# Guardar el modelo en formato pkl
joblib.dump(rus_rf, 'rus_random_forest.pkl')