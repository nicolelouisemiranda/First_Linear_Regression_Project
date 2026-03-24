# ******************************************************************************
#                         LINEAR REGRESSION BY HAND
# ******************************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
#import plotly.express as px

# import train data
train_data = pd.read_csv('train_energy_data.csv')

# summary of columns, data types and null values
print('****** TRAIN DATA INFO ******')
print(train_data.info())

# check for duplicated rows
print('duplicated rows:', train_data.duplicated().sum())

# remove columns that will not be used
train_data = train_data.drop(columns=['Building Type', 'Day of Week', 'Number of Occupants', 'Average Temperature', 'Appliances Used'])

# checking how many data points we have
print('number of data points:', len(train_data))
print()

# is the data in a gaussian distribution?
g = sns.displot(train_data['Square Footage'], kind='hist', kde=True)
g.set_axis_labels("Área (m²)", "Contagens")
g.set(title="Distribuição da Área")
plt.tight_layout()
plt.savefig('plot_area_distribution.png')
plt.show()

g = sns.displot(train_data['Energy Consumption'], kind='hist', kde=True)
g.set_axis_labels("Consumo de Energia (kW)", "Contagens") 
g.set(title="Distribuição do Consumo de Energia")
plt.tight_layout()
plt.savefig('plot_energy_consumption_distribution.png')
plt.show()

# visualize the data
plt.scatter(train_data['Square Footage'], train_data['Energy Consumption'], s=5)
plt.xlabel('Área (m²)')
plt.ylabel('Consumo de Energia (kW)')
plt.title('Consumo de Energia vs Área')
plt.tight_layout()
plt.savefig('plot_energy_vs_area.png')
plt.show()

# is there a linear relationship between the variables?
pearson_coef, _ = stats.pearsonr(train_data['Square Footage'], train_data['Energy Consumption'])
print('****** CORRELATION ANALYSIS ******')
print("Pearson Correlation Coefficient:", f"{pearson_coef:.3f}")
print()

# ******************************************************************************
#                             FUNCTIONS USED
# ******************************************************************************
# normalize data
def normalize_data(arr):
  return (arr - arr.min()) / (arr.max() - arr.min())
  #array_normalized = []
  #for item in arr:
  #  valor_normalized = (item - arr.min()) / (arr.max() - arr.min())
  #  array_normalized.append(valor_normalized)
  #return array_normalized

# linear function
def hipotesis(coef_0, coef_1, x):
  return coef_0 + coef_1 * x

# cost function
def f_cost(arr, coef_0, coef_1, y):
  sum = 0
  m = len(arr)
  for i in range(len(arr)):
    sum += (hipotesis(coef_0, coef_1, arr[i]) - y[i])**2
  return (1/m) * sum

def linear_regression(x_norm, y_norm, alpha=0.01, a_0=0, a_1=0):
  contador = 1
  J = 0
  J_arr = []
  a_0_arr = []
  a_1_arr = []
  residuo_arr = []

  while contador >= 0.00001:
    # calculate cost function and update contador
    J = f_cost(x_norm, a_0, a_1, y_norm)
    J_arr.append(J)

    # calculate the gradient
    soma_0 = 0
    soma_1 = 0
    for i in range(len(x_norm)):
      soma_0 += (hipotesis(a_0, a_1, x_norm[i]) - y_norm[i])
      soma_1 += (hipotesis(a_0, a_1, x_norm[i]) - y_norm[i]) * x_norm[i]
    gradiente_0 = soma_0 / len(x_norm)
    gradiente_1 = soma_1 / len(x_norm)

    # calculate the new value for the coefficients
    a_0 = a_0 - alpha * gradiente_0
    a_1 = a_1 - alpha * gradiente_1

    # add the coefficients in an array
    a_0_arr.append(a_0)
    a_1_arr.append(a_1)

    # calculate new cost function and update
    J_new = f_cost(x_norm, a_0, a_1, y_norm)
    contador = J - J_new
    J = J_new

  # calculando o residuo
  for i in range(len(x_norm)):
    residuo = y_norm[i] - hipotesis(a_0, a_1, x_norm[i])
    residuo_arr.append(residuo)

  return a_0, a_1, a_0_arr, a_1_arr, residuo_arr, J_arr

# de-normalize the coeficients after the linear regression
def denormalizer(x, y, coef_0, coef_1):
  deltay = y.max() - y.min()
  deltax = x.max() - x.min()
  coef0_desn = -(deltay/deltax) * coef_1 * x.min() + (coef_0 * deltay) + y.min()
  coef1_desn = (deltay/deltax) * coef_1
  return coef0_desn, coef1_desn

# ******************************************************************************
#                  APPLYING LINEAR REGRESSION ON THE DATA
# ******************************************************************************
# normalize the data
x_n = normalize_data(train_data['Square Footage'])
y_n = normalize_data(train_data['Energy Consumption'])

# apply linear regression function
coef0, coef1, coef0_arr, coef1_arr, res, j = linear_regression(x_norm=x_n, y_norm=y_n, alpha=0.01, a_0=0.8, a_1=0.8)

# de-normalize the coefficients
coef0_denormalized, coef1_denormalized = denormalizer(x=train_data['Square Footage'], y=train_data['Energy Consumption'], coef_0 = coef0, coef_1=coef1)

print('****** RESULTS ******')
print('Coeficiente a_0:', f"{coef0:.3f}")
print('Coeficiente a_1:', f"{coef1:.3f}")
print()

print('****** DENORMALIZED COEFFICIENTS ******')
print('Coeficiente a_0 desnormalizado:', f"{coef0_denormalized:.3f}")
print('Coeficiente a_1 desnormalizado:', f"{coef1_denormalized:.3f}")
print()

# check residuals for normality
g = sns.displot(res, kind='hist', kde=True)
g.set_axis_labels("Resíduo", "Densidade")
g.set(title="Distribuição dos Resíduos")
plt.tight_layout()
plt.savefig('plot_residuals_distribution.png')
plt.show()

stats.probplot(res, dist="norm", plot=plt)
plt.title('QQ Plot dos Resíduos')
plt.xlabel('Quantis Teóricos')
plt.ylabel('Quantis Observados')
plt.tight_layout()
plt.savefig('plot_qq_residuals_probplot.png')
plt.show()

# ******************************************************************************
#                                PLOTS
# ******************************************************************************

fig = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)

# cost function plot
plt.scatter(coef0_arr, j, label='Coeficiente $a_0$', c='blue', s=5)
plt.scatter(coef1_arr, j, label='Coeficiente $a_1$', c='black', s=5)
plt.xlabel('Valor dos Coeficientes')
plt.ylabel('Função Custo')
plt.title('Evolução da Função Custo')
plt.legend()

# grafico do residuo
plt.subplot(2,2,2)
plt.plot(hipotesis(coef0, coef1, train_data['Square Footage']), [0]*len(train_data['Square Footage']), c='black')
plt.scatter(hipotesis(coef0, coef1, train_data['Square Footage']), res, s=5)
plt.xlabel('Valores de y(x) Retornados pelo Modelo')
plt.title('Variabilidade dos Resíduos')
plt.ylabel('Residuo')

# ajuste
y_fit = []
for i in x_n:
  y_fit.append(hipotesis(coef0, coef1, i))

plt.subplot(2,2,3)
plt.scatter(x_n, y_n, label='Dados Normalizados', s=5)
plt.plot(x_n, y_fit, label='Regressão Linear', c='red')
plt.xlabel('Área')
plt.ylabel('Consumo de Energia')
plt.title('Ajuste nos Dados Normalizados')

# linear function with de-normalized coefficients
y_desn = []
for i in train_data['Square Footage']:
  y_desn.append(hipotesis(coef0_denormalized, coef1_denormalized, i))

plt.subplot(2,2,4)
plt.scatter(train_data['Square Footage'], train_data['Energy Consumption'], s=5)
plt.plot(train_data['Square Footage'], y_desn, label='Regressão Linear', c='red')
plt.xlabel('Área')
plt.ylabel('Consumo de Energia')
plt.title('Ajuste nos Dados Reais')

plt.tight_layout()
plt.savefig('plot_linear_regression.png')
plt.show()

# ******************************************************************************
#                        FIT MODEL TO TEST DATA
# ******************************************************************************
print('****** PERFORM TEST ******')
# import test data
test_data = pd.read_csv('test_energy_data.csv')

# remove columns that will not be used
test_data = test_data.drop(columns=['Building Type', 'Day of Week', 'Number of Occupants', 'Average Temperature', 'Appliances Used'])

# checking how many data points we have
print('number of data points in test data:', len(test_data))
print()

# fit the model to the test data
y_test = []
for x in test_data['Square Footage']:
  y_test.append(hipotesis(coef0_denormalized, coef1_denormalized, x))

# plot the fit on the test data
plt.scatter(test_data['Square Footage'], test_data['Energy Consumption'], s=5)
plt.plot(test_data['Square Footage'], y_test, label='Regressão Linear', c='red')
plt.xlabel('Área')
plt.ylabel('Consumo de Energia')
plt.title('Ajuste nos Dados de Teste')
plt.tight_layout()
plt.savefig('plot_linear_regression_test_data.png')
plt.show()

# evaluate model
# R² (coefficient of determination)
# R² is a measure of how well the model fits the data, and it ranges from 0 to 1. 
# A value of 0 indicates that the model does not explain any of the variability in the data, while a value of 1 indicates that the model explains all of the variability in the data.
# this coeficient has some limitations, as it can be affected by outliers or non-linear relationships between the variables.   
 
def R_2(y_t, y_p):
  y_mean = np.mean(y_t)
  ss_residual = sum((y_t - y_p) ** 2)
  ss_total = sum((y_t - y_mean) ** 2)
  
  r_squared = 1 - (ss_residual / ss_total)
  return r_squared

R_coef = R_2(test_data['Energy Consumption'], y_test)
print('my R²:', f"{R_coef:.3f}")

r2 = r2_score(test_data['Energy Consumption'], y_test)
print('sklearn R²:', f"{r2:.3f}")
print()

# mean absolute error (MAE)

# outliers dont impact as much in MAE
def MAE(y_t, y_p):
  mae = np.mean(np.abs(y_t - y_p))
  return mae

MAE = MAE(test_data['Energy Consumption'], y_test)
print('my MAE:', f"{MAE:.3f}")

MAE_sklearn = mean_absolute_error(test_data['Energy Consumption'], y_test)
print('sklearn MAE:', f"{MAE_sklearn:.3f}")
print()

# RMSE (Root Mean Squared Error)
def RMSE(y_t, y_p):
  rmse = np.sqrt(np.mean((y_t - y_p) ** 2))
  return rmse

RMSE = RMSE(test_data['Energy Consumption'], y_test)
print('my RMSE:', f"{RMSE:.3f}")

RMSE_sklearn = root_mean_squared_error(test_data['Energy Consumption'], y_test)
print('sklearn RMSE:', f"{RMSE_sklearn:.3f}")
print()

# ******************************************************************************
#                    COMPARISON WITH SKLEARN LINEAR REGRESSION
# ******************************************************************************
print('****** PERFORM TEST WITH SKLEARN ******')
X_train = train_data['Square Footage']
y_train = train_data['Energy Consumption']

X_test = test_data['Square Footage']
y_test = test_data['Energy Consumption']

# normalize data
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train.values.reshape(-1, 1))
y_train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

X_test_scaled = scaler.transform(X_test.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
# initiate the model 
model = LinearRegression()

# fit model to training data
model.fit(X_train_scaled, y_train_scaled)

# coeficients returned
coef0_sklearn = model.intercept_[0]
coef1_sklearn = model.coef_[0][0]

print('Coeficiente a_0 sklearn:', f"{coef0_sklearn:.3f}")
print('Coeficiente a_1 sklearn:', f"{coef1_sklearn:.3f}")
print()

# predict the values of y using the model
y_predict = model.predict(X_test_scaled)

#R²
r2_sklearn = r2_score(y_test_scaled, y_predict)
print('R² sklearn:', f"{r2_sklearn:.3f}")

#MAE
MAE_sklearn = mean_absolute_error(y_test_scaled, y_predict)
print('MAE sklearn:', f"{MAE_sklearn:.3f}")

#RMSE 
RMSE_sklearn = root_mean_squared_error(y_test_scaled, y_predict)
print('RMSE sklearn:', f"{RMSE_sklearn:.3f}")

