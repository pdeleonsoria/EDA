from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import missingno as msno 
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.model_selection import train_test_split


datos = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv")
#datos.head()



#datos.info()
#datos.nunique()

#Con datos.info() ya veo que hay varias columnas con Nans: name, host_name, last_review y reviews_per_month.

datos_util = datos.drop(["name", "host_id", "latitude", "longitude", "host_name", "last_review", "reviews_per_month", "calculated_host_listings_count"], axis = 1)

#datos_util.info()

#Como todas las columnas tienen 48895 entradas, no tengo que revisar que hayan valores vacíos, pero lo compruebo:

msno.matrix(datos_util)

#Confirmado 

datos_util.nunique()
#Representación de variables no numéricas 

fig, axis = plt.subplots(2, 2, figsize=(15, 7))


sns.histplot(ax = axis[0,0], data = datos_util, x = "neighbourhood_group")
sns.histplot(ax = axis[0,1], data = datos_util, x = "neighbourhood").set_xticks([])
sns.histplot(ax = axis[1,0], data = datos_util, x = "room_type")
sns.histplot(ax = axis[1,1], data = datos_util, x = "availability_365")
plt.show()

#Representación de variables numéricas 
datos_util["log_price"]=np.log(datos.price)
datos_util["log_minn"]=np.log(datos.minimum_nights)


fig, axis = plt.subplots(2, 2, figsize = (15, 7))

sns.histplot(ax = axis[0, 0], data = datos_util, x = "log_price")
sns.boxplot(ax = axis[1, 0], data = datos_util, x = "log_price")

sns.histplot(ax = axis[0, 1], data = datos_util, x = "log_minn")
sns.boxplot(ax = axis[1, 1], data = datos_util, x = "log_minn")


plt.show()

#Análisis de bivariables:

#Numéricas:

fig, axis = plt.subplots(4, 2, figsize = (10, 16))


sns.regplot(ax = axis[0, 0], data = datos_util, x = "minimum_nights", y = "price")
sns.heatmap(datos_util[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = datos_util, x = "number_of_reviews", y = "price").set(ylabel = None)
sns.heatmap(datos_util[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = datos_util, x = "minimum_nights", y = "number_of_reviews").set(ylabel = None)
sns.heatmap(datos_util[["number_of_reviews", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[3, 0]).set(ylabel = None)
fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])


plt.tight_layout()


plt.show()

#No existe correlación precio-noches mínimas, precio-reviews ni tampoco entre noches mínimas-reviews

#Categóricas:

fig, axis = plt.subplots(figsize = (7, 5))
sns.countplot(data = datos_util, x = "room_type", hue = "neighbourhood_group")
plt.show()
#Manhathan presenta la mayoría de alojamientos, seguido por Brooklyn. Staten Island apenas tiene oferta. El tipo de alojamiento mayoritariamente ofrecido es de apartamento completo y su máxima representación está en Manhattan, seguido de habitación privada en Brooklyn

plt.figure(figsize=(7, 5))
sns.barplot(x="neighbourhood_group", y="availability_365", data=datos_util)
plt.title("Disponibilidad promedio al año por Neighbourhood Group")
plt.ylabel("Disponibilidad (días)")
plt.xlabel("Neighbourhood Group")
plt.show()
#El barrio con mayor disponibilidad media al año es Staten Island con más de 200 noches, seguida por el Bronx y en último lugar Manhattan con menos de 100 noches.


#Análisis de todas las variables por mapa de calor:

datos_util["room_type"] = pd.factorize(datos_util["room_type"])[0]
datos_util["neighbourhood_group"] = pd.factorize(datos_util["neighbourhood_group"])[0]
datos_util["neighbourhood"] = pd.factorize(datos_util["neighbourhood"])[0]

fig, axes = plt.subplots(figsize=(15, 15))
sns.heatmap(datos_util[["neighbourhood_group", "neighbourhood", "room_type", "price", "minimum_nights",	"number_of_reviews", "availability_365"]].corr(), annot = True, fmt = ".2f")
plt.tight_layout()
plt.show()

#No existe correlación fuerte entre ninguna variable

#Análisis de todas las variables a la vez:
sns.pairplot(data = datos_util)


#Detecctión de outliers y eliminación por rango de intercuartiles:
minn_est = datos_util["minimum_nights"].describe()
print(minn_est)

nights_iq = minn_est["75%"] - minn_est["25%"]

slimit_minn = minn_est["75%"] + 1.5 * nights_iq
ilimit_minn = minn_est["25%"] - 1.5 * nights_iq

print(f"Los valores para noches mínimas son:{slimit_minn} y {ilimit_minn}")

price_est = datos_util["price"].describe()
print(price_est)

price_iq = price_est["75%"] - price_est["25%"]

slimit_price = price_est["75%"] + 1.5 * price_iq
ilimit_price = price_est["25%"] - 1.5 * price_iq

print(f"Los valores para precio son:{slimit_price} y {ilimit_price}")

number_of_reviews_est = datos_util["number_of_reviews"].describe()
print(number_of_reviews_est)

number_of_reviews_iq = number_of_reviews_est["75%"] - number_of_reviews_est["25%"]

slimit_number_of_reviews = number_of_reviews_est["75%"] + 1.5 * number_of_reviews_iq
ilimit_number_of_reviews = number_of_reviews_est["25%"] - 1.5 * number_of_reviews_iq

print(f"Los valores para reviews son:{slimit_number_of_reviews} y {ilimit_number_of_reviews}")

#Solo vamos a utilizar los límites superiores porque los inferiores son siempre negativos y no tienen sentido:
#Eliminamlos los outlayers:

datos_util = datos_util[datos_util["minimum_nights"] <= 11]
datos_util = datos_util[datos_util["price"] <= 334]
datos_util = datos_util[datos_util["number_of_reviews"] <= 58.5]

#Normalización de variables que afectan al precio:


datos_util["neighbourhood_group"] = pd.factorize(datos_util["neighbourhood_group"])[0]
datos_util["room_type"] = pd.factorize(datos_util["room_type"])[0]

num_variables = ["number_of_reviews", "minimum_nights", "availability_365", "neighbourhood_group", "room_type"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(datos_util[num_variables])
df_scal = pd.DataFrame(scal_features, index = datos_util.index, columns = num_variables)
df_scal["price"] = datos_util["price"]
df_scal.head()


#Train y test

X = df_scal.drop("price", axis=1)
y = df_scal["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selection_model = SelectKBest(score_func=f_regression)  
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()

X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=X_test.columns.values[ix])

X_train_sel.head()

#Guardar (Esto la verdad lo he hecho porque lo he visto en la solución)

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)