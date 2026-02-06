import numpy as np
from shapely.geometry import Polygon, Point
from shapely.ops import cascaded_union
def load_contours_from_file(filename):
    """
    Загружает контуры из текстового файла
    Формат файла: 
    X1 Y1
    X2 Y2
    ...
    (пустая строка между контурами)
    """
    contours = []
    current_contour = []
    
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                if current_contour:
                    contours.append(current_contour)
                    current_contour = []
            else:
                try:
                    x, y = map(float, line.split())
                    current_contour.append((x, y))
                except:
                    raise ValueError(f"Ошибка в строке: {line}")
    
    if current_contour:
        contours.append(current_contour)
        
    return contours
def Sample2 (outer_contour, inner_contour, step):
    """
    Генерирует точки внутри области, ограниченной двумя контурами
    
    Параметры:
    outer_contour - внешний контур (список координат [(x1, y1), (x2, y2), ...])
    inner_contour - внутренний контур (список координат [(x1, y1), (x2, y2), ...])
    step - шаг между точками
    """
    
    # Создаем полигоны из контуров
    outer_poly = Polygon(outer_contour)
    inner_poly = Polygon(inner_contour)
    
    # Создаем область между контурами
    area = outer_poly.difference(inner_poly)
    
    # Получаем границы области
    minx, miny, maxx, maxy = area.bounds
    
    # Генерируем точки
    points = []
    for x in np.arange(minx, maxx, step):
        for y in np.arange(miny, maxy, step):
            point = Point(x, y)
            if area.contains(point):
                points.append((x, y))
                
    return points

def Sampler():
    # Пример внешнего контура (прямоугольник)
    outer_contour = [
        (0, 0),
        (10, 0),
        (10, 10),
        (0, 10)
    ]
    
    # Пример внутреннего контура (квадрат)
    inner_contour = [
        (2, 2),
        (8, 2),
        (8, 8),
        (2, 8)
    ]
    
    step = 1.0  # Шаг между точками
    
    points = Sample2(outer_contour, inner_contour, step)
    
    # Выводим результат
    print(f"Количество точек: {len(points)}")
    for point in points:
        print(f"Точка: {point}")

if __name__ == "__main__":
    Sampler()
