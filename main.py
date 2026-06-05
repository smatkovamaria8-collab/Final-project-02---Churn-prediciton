import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
import keras_tuner
from keras import callbacks
from keras.models import load_model
import streamlit as st
import tensorflow as tf

sns.set_style("whitegrid")

data = pd.read_csv("internet_service_churn.csv")
print(data.head(10))
data = data.iloc[:, 1:]
data = data.drop_duplicates().reset_index(drop=True)

# sns.countplot(x="is_tv_subscriber", hue="is_tv_subscriber", data=data, legend= True)
# plt.xlabel("Підписка клієнтів на телебачення", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.legend(["0 - без підписки", "1 - з підпискою"], fontsize=10)
# plt.show()

# sns.countplot(x="is_movie_package_subscriber", hue="is_movie_package_subscriber", data=data, legend= True)
# plt.xlabel("Підписка клієнтів на пакет фільмів", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.legend(["0 - без підписки", "1 - з підпискою"], fontsize=10)
# plt.show()

# sns.histplot(x = "subscription_age", data=data)
# plt.xlabel("Термін підписки", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.show()

# sns.histplot(x = "bill_avg", data=data)
# plt.xlabel("Середній рахунок", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.xlim(-10, 120) 
# plt.show()

# sns.histplot(x = "reamining_contract", data=data)
# plt.xlabel("Термін контракту, що залишився", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.show()

# sns.countplot(x="service_failure_count", data=data)
# plt.xlabel("Кількість збоїв у сервісі", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.show()

# sns.histplot(x = "download_avg", data=data)
# plt.xlabel("Середня кількість завантажень", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.xlim(0, 600) 
# plt.show()

# sns.histplot(x = "upload_avg", data=data)
# plt.xlabel("Середня кількість вивантажень", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.xlim(0, 50) 
# plt.show()

# sns.countplot(x="download_over_limit", data=data)
# plt.xlabel("Скачувань понад норму", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.show()

# sns.countplot(x="churn", hue = "churn", data=data, legend=True)
# plt.xlabel("Дані про відток", fontsize="medium", color="midnightblue")
# plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
# plt.legend(["0 - клієнт залишився", "1 - клієнт пішов"], fontsize=10)
# plt.show()

# print(data.info())


# data.describe()

# correlation = data.corr()
# plt.figure(figsize=(12,12))
# sns.heatmap(correlation, annot=True, cmap="coolwarm")
# plt.title("Матриця кореляції")
# plt.show()

# print(data.isnull().sum())

columns = ["reamining_contract","download_avg","upload_avg"]
for column in columns:
    data[column] = data[column].fillna(data[column].median())

# print(data.isnull().sum())

data = data.astype(float)
print(data.info())


data = shuffle(data, random_state=42)

scaler = StandardScaler().set_output(transform="pandas")

X = data.iloc[:, :-1]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

### Звичайний класифікатор для порівння значень моделі


# sgd_clf = SGDClassifier(random_state=42)

# param_grid = {
#     'penalty': ['l1', 'l2', 'elasticnet'],
#     'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
# }

# grid_search = GridSearchCV(estimator=sgd_clf, param_grid=param_grid, cv=5, scoring='accuracy')

# grid_search.fit(X_train, y_train)

# print(f"Найкращі параметри: {grid_search.best_params_}")

# best_model = grid_search.best_estimator_
# y_pred = best_model.predict(X_test)

# print(classification_report(y_test, y_pred))


### Нейронна мережа для порівняння якості моделі

# callback = callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)

# def build_model(hp):
#     model = Sequential()
#     for i in range(hp.Int("num_layers", 1, 3)):

#         model.add(Dense(
#             units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
#             activation=hp.Choice(f"activation_{i}", ["relu", "tanh"])))
        
#         if hp.Boolean("dropout"):
#             model.add(Dropout(rate=0.25))
    
#     model.add(Dense(1, activation="sigmoid"))

#     learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
#     model.compile(
#         optimizer=Adam(learning_rate=learning_rate),
#         loss="binary_crossentropy",
#         metrics=["accuracy"],
#     )
#     return model


# tuner = keras_tuner.RandomSearch(
#     hypermodel=build_model,
#     objective="val_accuracy",
#     max_trials=15,
#     executions_per_trial=2,
#     overwrite=True,
#     directory="my_dir",
#     project_name="neuralnet",
# )


# tuner.search(X_train, y_train, epochs=20, validation_split=0.2, callbacks = [callback])

# best_models = tuner.get_best_models(num_models=1)
# best_model = best_models[0]
# best_model.summary()
# best_model.save("my_best_model.keras")

# y_pred = best_model.predict(X_test)

# y_pred = (y_pred > 0.5).astype("int")

# print(classification_report(y_test, y_pred))

### Нейронна модель вказала на кращі результати, тому для подальшої класифікації даних будемо використовувати її

loaded_model = load_model("my_best_model.keras")

st.title("Прогнозування ймовірності відтоку клієнта")

st.write("Будь ласка, введіть дані про клієнта обираючи відповідь із наявних")

is_tv_sub = st.radio("Чи має підписку на телебачення", ["Ні", "Так"])
is_movie_sub = st.radio("Чи має підписку на пакет фільмів", ["Ні", "Так"])

st.write("Будь ласка, введіть дані про клієнта використовуючи поля для вводу (просимо писати лише числа)")

sub_age = st.number_input("Вкажіть термін підписки", min_value= 0.0, value= None, step=0.01)
bill_sum = st.number_input("Вкажіть його середній рахунок", min_value= 0.0, value= None, step=0.01)
left_contract = st.number_input("Вкажіть залишковий термін дії контракту", min_value= 0.0, value= None, step=0.01)
num_errors = st.number_input("Вкажіть кількість збоїв у сервері", min_value= 0.0, value= None, step=0.01)
num_downl = st.number_input("Вкажіть середню кількість завантажень", min_value= 0.0, value= None, step=0.01)
num_upload = st.number_input("Вкажіть середню кількість вивантажень", min_value= 0.0, value= None, step=0.01)
downl_overlim = st.number_input("Вкажіть кількість скачувань понад норму", min_value= 0.0, value= None, step=0.01)

coded_words = {
    "Ні" : 0,
    "Так" : 1}

encoded_tv_sub = coded_words[is_tv_sub]
encoded_movie_sub = coded_words[is_movie_sub]

all_variables = [encoded_tv_sub, encoded_movie_sub, sub_age, bill_sum, left_contract, num_errors,
                 num_downl, num_upload, downl_overlim]

if st.button("Спрогнозувати відтік"):
    if None in all_variables:
        st.warning("Ви не вказали повністю всі змінні, будь ласка введіть пропущені")
    else:
        st.success("Всі дані були успішно внесені")
        client_array = np.array([all_variables], dtype = float)

        prediction = loaded_model.predict(client_array)
        prediction = float(prediction[0][0])
        leave_prediction = prediction * 100
        stay_prediction = 100 - leave_prediction

        st.write(f"Результат прогнозу:\n"
                 f"{leave_prediction:.2f}% - клієнт піде\n"
                 f"{stay_prediction:.2f}% - клієнт залишиться")

        pred_plot = [1 - prediction, prediction]
        labels = ["Залишиться", "Піде"]
        
        plt.clf()
        sns.barplot(x = labels, y = pred_plot, palette = ["mediumseagreen", "crimson"])
        plt.xlabel("Ймовірний статус", fontsize="medium", color="midnightblue")
        plt.ylabel("Ймовірність", fontsize="medium", color="midnightblue")
        plt.title("Результати прогнозування клієнта", fontsize="medium", color="midnightblue")
        st.pyplot(plt)
