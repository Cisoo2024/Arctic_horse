import pandas as pd
import numpy as np
from itertools import permutations
from ships_rout_table import speed_by_class, arc7_ships, arc9_ships

# Загрузка данных из файла
df = pd.read_excel('IntegrVelocity.xlsx')


# Функция для расчета времени прохождения маршрута
def calculate_time(route, ice_class):
    total_time = 0
    for i in range(len(route) - 1):
        start = route[i]
        end = route[i + 1]
        time = df.loc[(df['Start'] == start) & (df['End'] == end) & (df['Ice Class'] == ice_class), 'Time'].values[0]
        total_time += time
    return total_time


# Функция для поиска наиболее быстрого маршрута
def find_fastest_route(ports, ice_class):
    # Генерируем все возможные маршруты
    routes = list(permutations(ports))

    # Вычисляем время прохождения для каждого маршрута
    times = [calculate_time(route, ice_class) for route in routes]

    # Находим индекс наиболее быстрого маршрута
    fastest_index = np.argmin(times)

    # Возвращаем наиболее быстрый маршрут
    return routes[fastest_index]


# Пример использования
ports = [1, 2, 3, 4, 5]
for ice_class in range(1, 10):
    fastest_route = find_fastest_route(ports, ice_class)
    print(f"Для ледового класса {ice_class} наиболее быстрый маршрут: {', '.join(map(str, fastest_route))}")
    print(f"Время прохождения: {calculate_time(fastest_route, ice_class)} часов")