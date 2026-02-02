# ******************************************************************************
#                EXAMPLE 1 - LINEAR REGRESSION BY HAND
# ******************************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# import train data
train_data = pd.read_csv('test_energy_data.csv')

# summary of columns, data types and null values
print(train_data.info())

# check for duplicated rows
print('duplicated rows:', train_data.duplicated().sum())

# data type correction
train_data['Building Type'] = train_data['Building Type'].astype('string')
train_data['Day of Week'] = train_data['Day of Week'].astype('string')

# ******************************************************************************
#                             FUNCTIONS USED
# ******************************************************************************
# normalize data
def normalize_data(arr):
  array_normalized = []
  for item in arr:
    valor_normalized = (item - arr.min()) / (arr.max() - arr.min())
    array_normalized.append(valor_normalized)
  return array_normalized

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
  coef0_desn = (deltay/deltax) * coef_1 * x.min() + (coef_0 * deltay) + y.min()
  coef1_desn = (deltay/deltax) * coef_1
  return coef0_desn, coef0_desn

# ******************************************************************************
#                  APPLYING LINEAR REGRESSION ON THE DATA
# ******************************************************************************
# normalize the data first
x_n = normalize_data(train_data['Square Footage'])
y_n = normalize_data(train_data['Energy Consumption'])

# apply linear regression function
coef0, coef1, coef0_arr, coef1_arr, res, j = linear_regression(x_norm=x_n, y_norm=y_n, alpha=0.01, a_0=1, a_1=1)

# de-normalize the coefficients
coef0_denormalized, coef1_denormalized = denormalizer(x=train_data['Square Footage'],y=train_data['Energy Consumption'],coef_0 = coef0,coef_1=coef1)

# check residuals for normality
g = sns.displot(res, kind='hist', kde=True)
g.set_axis_labels("Resíduo", "Densidade")
g.set(title="Distribuição dos Resíduos")
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
for i in x_n:
  y_desn.append(hipotesis(coef0_denormalized, coef1_denormalized, i))

plt.subplot(2,2,4)
plt.scatter(train_data['Square Footage'], train_data['Energy Consumption'], s=5)
plt.plot(train_data['Square Footage'], y_desn, label='Regressão Linear', c='red')
plt.xlabel('Área')
plt.ylabel('Consumo de Energia')
plt.title('Ajuste nos Dados Reais')

plt.tight_layout()
#plt.savefig('/content/plot_linear_regression.png')
plt.show()