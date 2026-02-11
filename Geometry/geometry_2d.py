# -*- coding: utf-8 -*-
import math
#Функция загрузки контура из файла
def FoilLoader(path):
    foil_coord = [] #Массив координат профиля
    try:
        with open(path, 'r') as foil_file:
            for line_number, line in enumerate(foil_file, start=0):
                line = line.strip()
                #print(line)
                if not line:
                    continue
                try:
                    x_str, y_str = line.split('\t')
                    x = float(x_str)
                    y = float(y_str)
                    foil_coord.append((x,y))
                except ValueError:
                    print (f"Ошибка в строке {line_number}: некорректный формат данных")
    except FileNotFoundError:
        print(f"Ошибка: файл по пути {path} не найден")
    except Exception as e:
        print(f"Произошла ошибка при чтении файла: {str(e)}")
    return foil_coord
def UnclosedDomen(foil_coord):
    import numpy as np
    Domen = []
    Domen_leight = float(input("Введите длину домена от носка контура, м: "))
    Domen_height = float(input("Введите высоту домена от верхней границы, м: "))
    #Domen_point1 = (-Domen_leight, Domen_height)
    Domen = [(float(np.min(foil_coord, axis=0)[0]), float(np.min(foil_coord, axis=0)[1])),
             (-Domen_leight, float(np.min(foil_coord, axis=0)[1])), 
             (-Domen_leight, Domen_height),
             (float(np.max(foil_coord, axis=0)[0]), Domen_height), 
             (float(np.max(foil_coord, axis=0)[0]), float(np.max(foil_coord, axis=0)[1]))]
    return Domen
def ClosedDomen(foil_coord):
    import numpy as np
    Domen = []
    Domen_leight1 = float(input("Введите длину домена перед профилем, м: "))
    Domen_leight2 = float(input("Введите длину домена за профилем, м: "))
    Domen_height1 = float(input("Введите высоту домена над профилем, м: "))
    Domen_height2 = float(input("Введите высоту домена под профилем, м: "))
    #Domen_point1 = (-Domen_leight, Domen_height)
    Domen = [(-Domen_leight1, -Domen_height2),
             (-Domen_leight1, Domen_height1),
             (Domen_leight2, Domen_height1),
             (Domen_leight2, -Domen_height2),
             (-Domen_leight1, -Domen_height2)]
    return Domen
def Plot(contour1, contour2, title="Общий вид домена", 
                  color1='blue', color2='red', 
                  label1='Профиль (стенка)', label2='Граница расчётной области'):
    """
    Функция для визуализации двух контуров на одном графике
    
    Параметры:
    contour1, contour2 - списки координат точек контуров [[x1,y1], [x2,y2], ...]
    title - заголовок графика
    color1, color2 - цвета контуров
    label1, label2 - метки для легенды
    """
    import matplotlib.pyplot as plt
    import numpy as np
    # Преобразуем входные данные в numpy массивы
    contour1 = np.array(contour1)
    contour2 = np.array(contour2)
    
    # Создаем график
    plt.figure(figsize=(8, 6))
    
    # Рисуем первый контур
    plt.plot(contour1[:, 0], contour1[:, 1], 
             marker='o', linestyle='-', 
             color=color1, label=label1)
    
    # Рисуем второй контур
    plt.plot(contour2[:, 0], contour2[:, 1], 
             marker='o', linestyle='-', 
             color=color2, label=label2)
    
    # Добавляем настройки графика
    plt.title(title)
    plt.xlabel('Ось X')
    plt.ylabel('Ось Y')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # Сохраняем пропорции осей
    plt.grid(True)
    plt.show()
def Saving(coordinates, filename):
    # Проверяем формат входных данных
    if not all(isinstance(coord, tuple) and len(coord) == 2 for coord in coordinates):
        raise ValueError("Координаты должны быть списком кортежей с двумя элементами")
    
    # Разделяем координаты на x и y
    x_coords = [str(point[0]) for point in coordinates]
    y_coords = [str(point[1]) for point in coordinates]
    
    # Открываем файл для записи
    with open(filename, 'w', encoding='utf-8') as file:
        # Записываем заголовок
        #file.write("X\tY\n")
        # Записываем данные построчно
        for x, y in zip(x_coords, y_coords):
            file.write(f"{x}\t{y}\n")
def g2d():
    print ("Укажите путь к файлу контура (path.../name.txt):")
    try:
        path = input("Путь файла: ")
        Foil = FoilLoader(path) #Загрузка профиля
        print ("Геометрия профиля загружена")
    except ValueError:
        print("Неверный путь файла")
    Closure = Foil[0] == Foil[-1] #Проверка замкнутости контура
    print ("Выполнена проверка замкнутости контура")
    if Closure == False: Domen = UnclosedDomen(Foil)
    else: Domen = ClosedDomen(Foil)
    Plot(Foil, Domen)
    Saving(Foil, "C:/Users/Ivan/source/repos/PINN_Flow2D/Geometry/Data/foil.txt")
    Saving(Domen, "C:/Users/Ivan/source/repos/PINN_Flow2D/Geometry/Data/domen.txt")
    print ("Расчётная область сохранена")


            