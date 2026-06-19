# Airfoil PINN UI Prototype

Автономный прототип интерфейса на PySide6. Он не импортирует основной код проекта и использует мок-данные, чтобы интерфейс можно было тестировать отдельно от backend.

Запуск из корня проекта:

```powershell
python -m ui.app
```

Если PySide6 не установлен:

```powershell
python -m pip install -r ui/requirements.txt
```

Экран `Geometry` работает в замкнутом frontend-контуре:

- читает `.txt`, `.dat`, `.csv`, `.npy` координаты без импорта `Geometry.py`;
- принимает drag-and-drop файла геометрии прямо в окно;
- принимает ручной ввод координат в формате `x y`, `x,y` или `x; y`;
- показывает профиль, расчётную область, статистику и вершины домена;
- строит домен для замкнутого и незамкнутого профиля по тем же параметрам (`front`, `rear`, `top`, `bottom`);
- сохраняет тестовые артефакты в `ui/mock_outputs/Geometry`, не в backend-папки `Geometry/Data` и `Geometry/Metadata`.
