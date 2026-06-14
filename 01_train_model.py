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
import tensorflow as tf
import joblib

sns.set_style("whitegrid")

#Завантаження даних, видалення дублікатів

data = pd.read_csv("internet_service_churn.csv")
print(data.head(10))
data = data.iloc[:, 1:]
data = data.drop_duplicates().reset_index(drop=True)

#Візуалізація кожної змінної

sns.countplot(x="is_tv_subscriber", hue="is_tv_subscriber", data=data, legend= True)
plt.xlabel("Підписка клієнтів на телебачення", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.legend(["0 - без підписки", "1 - з підпискою"], fontsize=10)
plt.title("Розподіл клієнтів за наявністю підписки на телебачення", fontsize="medium", color="midnightblue")
plt.show()

sns.countplot(x="is_movie_package_subscriber", hue="is_movie_package_subscriber", data=data, legend= True)
plt.xlabel("Підписка клієнтів на пакет фільмів", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.legend(["0 - без підписки", "1 - з підпискою"], fontsize=10)
plt.title("Розподіл клієнтів за наявністю підписки на пакет фільмів", fontsize="medium", color="midnightblue")
plt.show()

sns.histplot(x = "subscription_age", data=data)
plt.xlabel("Термін підписки", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.title("Розподіл клієнтів за терміном підписки", fontsize="medium", color="midnightblue")
plt.show()

sns.histplot(x = "bill_avg", data=data)
plt.xlabel("Середній рахунок", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.xlim(-10, 120) 
plt.title("Розподіл клієнтів за середнім рахунком за користування сервісами компанії", fontsize="medium", color="midnightblue")
plt.show()

sns.histplot(x = "reamining_contract", data=data)
plt.xlabel("Термін контракту, що залишився", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.title("Розподіл клієнтів за терміном контракту, що залишився", fontsize="medium", color="midnightblue")
plt.show()

sns.countplot(x="service_failure_count", data=data)
plt.xlabel("Кількість збоїв у сервісі", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.title("Розподіл клієнтів за кількістю збоїв у сервісі", fontsize="medium", color="midnightblue")
plt.show()

sns.histplot(x = "download_avg", data=data)
plt.xlabel("Середня кількість завантажень", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.title("Розподіл клієнтів за середньою кількістю завантажень", fontsize="medium", color="midnightblue")
plt.xlim(0, 600) 
plt.show()

sns.histplot(x = "upload_avg", data=data)
plt.xlabel("Середня кількість вивантажень", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.title("Розподіл клієнтів за середньою кількістю вивантажень", fontsize="medium", color="midnightblue")
plt.xlim(0, 50) 
plt.show()

sns.countplot(x="download_over_limit", data=data)
plt.xlabel("Скачувань понад норму", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.title("Розподіл клієнтів за кількістю скачувань понад норму", fontsize="medium", color="midnightblue")
plt.show()

sns.countplot(x="churn", hue = "churn", data=data, legend=True)
plt.xlabel("Дані про відток", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість людей", fontsize="medium", color="midnightblue")
plt.legend(["0 - клієнт залишився", "1 - клієнт пішов"], fontsize=10)
plt.title("Розподіл клієнтів за відтоком", fontsize="medium", color="midnightblue")
plt.show()

print(data.info())

#Візуалізація описової статистики даних

data.describe()

#Проведення кореляції між всіма змінними

correlation = data.corr()
plt.figure(figsize=(12,12))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Матриця кореляції")
plt.show()

#Заповнення даних, що були пропущені, середніми значеннями (медіаною)

print(data.isnull().sum())

columns = ["reamining_contract","download_avg","upload_avg"]
for column in columns:
    data[column] = data[column].fillna(data[column].median())

print(data.isnull().sum())

data = data.astype(float)
print(data.info())

#Перемішування даних, їх стандартизація та поділ на тестові й навчальні дані

data = shuffle(data, random_state=42)

scaler = StandardScaler().set_output(transform="pandas")

X = data.iloc[:, :-1]
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")

# Проведення звичайного класифікатора для порівння значень якості моделі

sgd_clf = SGDClassifier(random_state=42)

param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet'],
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
}

grid_search = GridSearchCV(estimator=sgd_clf, param_grid=param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print(f"Найкращі параметри: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print(classification_report(y_test, y_pred))


## Нейронна мережа для порівняння значень якості моделі
## Тренування нейронної мережі використовуючи крос-валідаційний підбір гіперпараметрів

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

loaded_model = load_model("my_best_model.keras")
y_pred = loaded_model.predict(X_test)
y_pred = (y_pred > 0.5).astype("int")
print(classification_report(y_test, y_pred))

## Нейронна мережа вказала на кращі результати, тому для подальшої класифікації даних будемо використовувати її
