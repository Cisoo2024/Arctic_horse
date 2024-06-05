import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from ships_rout_table import speed_by_class, arc7_ships, arc9_ships


# 1. Сбор и подготовка данных
# Загрузка Excel файла
file_path = '/content/IntegrVelocity.xlsx'
xls = pd.ExcelFile(file_path)

# Чтение данных долготы и широты
lon_data = pd.read_excel(xls, 'lon').values
lat_data = pd.read_excel(xls, 'lat').values

# Список для хранения всех листов со скоростью и их данных
sheets_data = {}
for sheet in xls.sheet_names:
    if sheet not in ['lon', 'lat']:
        sheets_data[sheet] = pd.read_excel(xls, sheet).values
# Инициализация списка для данных
data = []
# Обходим каждый столбец данных по строкам и столбцам
n_rows, n_cols = lon_data.shape
for i in range(n_rows):
    for j in range(n_cols):
        # Извлекаем координаты для текущей ячейки
        lon = lon_data[i, j]
        lat = lat_data[i, j]
        # Собираем скорости для всех дат для текущей ячейки
        speeds = {sheet: sheets_data[sheet][i, j] for sheet in sheets_data}
        # Добавляем полную запись (координаты + скорости) в список
        data.append({'Долгота': lon, 'Широта': lat, **speeds})

# Преобразование списка словарей в DataFrame
final_speed_data = pd.DataFrame(data)

# Сохранение итоговых данных в новый Excel файл
output_path = '/content/Final_IntegrVelocity.xlsx'
final_speed_data.to_excel(output_path, index=False)

print("Обработка завершена и данные сохранены в:", output_path)

data = {
    'ice_class': [1, 2, 1, 3, 2],  # Ледовый класс
    'speed_in_clear_water': [20, 18, 22, 15, 19],  # Скорость в чистой воде (узлы)
    'distance': [100, 150, 200, 120, 180],  # Расстояние между пунктами (км)
    'ice_conditions': [3, 2, 4, 1, 3],  # Ледовые условия (баллы)
    'time_one_ship': [5, 6.5, 7.5, 4.8, 6.2],  # Время прохождения для одного судна (часы)
    'time_two_ships': [6, 7.8, 9.0, 5.9, 7.4],  # Время прохождения для двух судов (часы)
    'time_three_ships': [7, 9.1, 10.5, 7.0, 8.6]  # Время прохождения для трех судов (часы)
}

df = pd.DataFrame(data)

# Предварительный анализ данных
print(df.describe())

# Проверка на пропуски
print(df.isnull().sum())

# Разделение данных на признаки (X) и целевые переменные (y)
X = df[['ice_class', 'speed_in_clear_water', 'distance', 'ice_conditions']]
y_one_ship = df['time_one_ship']
y_two_ships = df['time_two_ships']
y_three_ships = df['time_three_ships']

# Разделение на обучающую и тестовую выборки
X_train_one, X_test_one, y_train_one, y_test_one = train_test_split(X, y_one_ship, test_size=0.2, random_state=42)
X_train_two, X_test_two, y_train_two, y_test_two = train_test_split(X, y_two_ships, test_size=0.2, random_state=42)
X_train_three, X_test_three, y_train_three, y_test_three = train_test_split(X, y_three_ships, test_size=0.2, random_state=42)

# Обучение модели
model_one_ship = LinearRegression()
model_two_ships = LinearRegression()
model_three_ships = LinearRegression()

model_one_ship.fit(X_train_one, y_train_one)
model_two_ships.fit(X_train_two, y_train_two)
model_three_ships.fit(X_train_three, y_train_three)

# Оценка качества модели
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

rmse_one, mae_one, r2_one = evaluate_model(model_one_ship, X_test_one, y_test_one)
rmse_two, mae_two, r2_two = evaluate_model(model_two_ships, X_test_two, y_test_two)
rmse_three, mae_three, r2_three = evaluate_model(model_three_ships, X_test_three, y_test_three)

print(f"Model for one ship: RMSE={rmse_one}, MAE={mae_one}, R^2={r2_one}")
print(f"Model for two ships: RMSE={rmse_two}, MAE={mae_two}, R^2={r2_two}")
print(f"Model for three ships: RMSE={rmse_three}, MAE={mae_three}, R^2={r2_three}")

# Использование модели для предсказания времени прохождения маршрута
new_data = pd.DataFrame({
    'ice_class': [2],
    'speed_in_clear_water': [20],
    'distance': [130],
    'ice_conditions': [3]
})

pred_time_one_ship = model_one_ship.predict(new_data)
pred_time_two_ships = model_two_ships.predict(new_data)
pred_time_three_ships = model_three_ships.predict(new_data)

print(f"Predicted time for one ship: {pred_time_one_ship[0]} hours")
print(f"Predicted time for two ships: {pred_time_two_ships[0]} hours")
print(f"Predicted time for three ships: {pred_time_three_ships[0]} hours")