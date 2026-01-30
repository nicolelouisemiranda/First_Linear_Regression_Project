# ******************************************************************************
#                EXAMPLE 1 - LINEAR REGRESSION BY HAND
# ******************************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# import train data
train_data = pd.read_csv('/content/train_energy_data.csv')

# data type correction
train_data['Building Type'] = train_data['Building Type'].astype('string')
train_data['Day of Week'] = train_data['Day of Week'].astype('string')