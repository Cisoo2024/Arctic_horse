# 1 - Добавим несколько различных алгоритмов регрессии
# для сравнения, таких как Decision Tree Regressor,
# Random Forest Regressor и Gradient Boosting Regressor\.
# 2 - Feature Engineering: - Добавим новые признаки
# (например, взаимодействие признаков\)\.
# - Нормализуем данные\.
# 3 - Оптимизация гиперпараметров:
# - Используем GridSearchCV для подбора оптимальных гиперпараметров\.
# 4 - Оценка моделей с использованием кросс\-валидации\*\*\.


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from ships_rout_table import speed_by_class, arc7_ships, arc9_ships

# 1. Сбор и подготовка данных
df = pd.read_excel('arctic_voyages.xlsx')
data = {
    'ice_class': [1, 2, 1, 3, 2],  # Ледовый класс
    'speed_in_clear_water': [20, 18, 22, 15, 19],  # Скорость в чистой воде (узлы)
    'distance': [100, 150, 200, 120, 180],  # Расстояние между пунктами (км)
    'ice_conditions': [3, 2, 4, 1, 3],  # Ледовые условия (баллы)
    'time_one_ship': [5, 6.5, 7.5, 4.8, 6.2],  # Время прохождения для одного судна (часы)
    'time_two_ships': [6, 7.8, 9.0, 5.9, 7.4],  # Время прохождения для двух судов (часы)
    'time_three_ships': [7, 9.1, 10.5, 7.0, 8.6]  # Время прохождения для трех судов (часы)
}
df_integr_veloc = pd.read_excel('IntegrVelocity.xlsx')


# Предварительный анализ данных
print(df.describe())

# Проверка на пропуски
print(df.isnull().sum())

# Разделение данных на признаки (X) и целевые переменные (y)
X = df[['ice_class', 'speed_in_clear_water', 'distance', 'ice_conditions']]
y_one_ship = df['time_one_ship']
y_two_ships = df['time_two_ships']
y_three_ships = df['time_three_ships']

# Feature Engineering: добавление полиномиальных признаков и нормализация
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Разделение на обучающую и тестовую выборки
X_train_one, X_test_one, y_train_one, y_test_one = train_test_split(X_poly_scaled, y_one_ship, test_size=0.2, random_state=42)
X_train_two, X_test_two, y_train_two, y_test_two = train_test_split(X_poly_scaled, y_two_ships, test_size=0.2, random_state=42)
X_train_three, X_test_three, y_train_three, y_test_three = train_test_split(X_poly_scaled, y_three_ships, test_size=0.2, random_state=42)

# Обучение моделей
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor()
}

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

for name, model in models.items():
    print(f"Evaluating {name} for one ship")
    rmse_one, mae_one, r2_one = evaluate_model(model, X_train_one, X_test_one, y_train_one, y_test_one)
    print(f"RMSE={rmse_one}, MAE={mae_one}, R^2={r2_one}")

    print(f"Evaluating {name} for two ships")
    rmse_two, mae_two, r2_two = evaluate_model(model, X_train_two, X_test_two, y_train_two, y_test_two)
    print(f"RMSE={rmse_two}, MAE={mae_two}, R^2={r2_two}")

    print(f"Evaluating {name} for three ships")
    rmse_three, mae_three, r2_three = evaluate_model(model, X_train_three, X_test_three, y_train_three, y_test_three)
    print(f"RMSE={rmse_three}, MAE={mae_three}, R^2={r2_three}")

# Оптимизация гиперпараметров с использованием GridSearchCV для RandomForest и GradientBoosting
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10], 'min_samples_split': [2, 5]
}

param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1],
    'max_depth': [3, 5]
}

grid_search_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=3)
grid_search_gb = GridSearchCV(GradientBoostingRegressor(), param_grid_gb, cv=3)

grid_search_rf.fit(X_train_one, y_train_one)
grid_search_gb.fit(X_train_one, y_train_one)

print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_)

# Оценка моделей с использованием кросс-валидации
best_rf_model = grid_search_rf.best_estimator_
best_gb_model = grid_search_gb.best_estimator_

rf_scores = cross_val_score(best_rf_model, X_poly_scaled, y_one_ship, cv=5)
gb_scores = cross_val_score(best_gb_model, X_poly_scaled, y_one_ship, cv=5)

print("Cross-validation scores for Random Forest:", rf_scores)
print("Cross-validation scores for Gradient Boosting:", gb_scores)

# Использование модели для предсказания времени прохождения маршрута
new_data = pd.DataFrame({
    'ice_class': [2],
    'speed_in_clear_water': [20],
    'distance': [130],
    'ice_conditions': [3]
})

new_data_poly = poly.transform(new_data)
new_data_scaled = scaler.transform(new_data_poly)

pred_time_one_ship_rf = best_rf_model.predict(new_data_scaled)
pred_time_one_ship_gb = best_gb_model.predict(new_data_scaled)

print(f"Predicted time for one ship with Random Forest: {pred_time_one_ship_rf[0]} hours")
print(f"Predicted time for one ship with Gradient Boosting: {pred_time_one_ship_gb[0]} hours")
