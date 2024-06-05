import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Загрузка данных в DataFrame
df = pd.read_excel('arctic_voyages.xlsx')
df_ports = pd.read_excel('GrafData.xlsx', sheet_name='edges', skiprows=1)
df_integr_veloc_lon = pd.read_excel('IntegrVelocity.xlsx', sheet_name='lon')
df_integr_veloc_lat = pd.read_excel('IntegrVelocity.xlsx', sheet_name='lat')
df_graf_data = pd.read_excel('GrafData.xlsx', sheet_name='points')

# Информация о датафрейме
print(df_ports.info())
print(df_integr_veloc_lon.info())
print(df_integr_veloc_lat.info())
print(df_graf_data.info())

# Описательная статистика
print(df_ports.describe())
print(df_integr_veloc_lon.describe())
print(df_integr_veloc_lat.describe())

# Обработка пропущенных значений

df_ports = df_ports.dropna(subset=['start_point_id', 'end_point_id'])
