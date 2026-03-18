import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class TGeometry:
    """
    Геометрия плоского T-образного тройника (бесконечного по оси Z).
    Задаётся размерами трёх прямоугольных каналов, сходящихся в начале координат (0,0).

    Атрибуты:
        L_in  : длина входного канала (влево по оси X)
        L_up  : длина верхнего выходного канала (вверх по оси Y)
        L_side: длина бокового выходного канала (вправо по оси X)
        W_in  : ширина входного канала (по оси Y)
        W_side: ширина бокового выходного канала (по оси Y)
    """
    name: str
    L_in: float
    L_up: float
    L_side: float
    W_in: float
    W_side: float

    def __post_init__(self):
        # Предварительный расчёт границ для ускорения проверок
        self.x_min = -self.W_in / 2
        self.x_max = self.L_side + self.W_in / 2
        self.y_min = -(self.L_in + self.W_side / 2) 
        self.y_max = self.L_up + self.W_side / 2

    def contains(self, x: float, y: float) -> bool:
        """
        Проверяет, принадлежит ли точка (x, y) внутренности тройника.
        Точка считается внутренней, если она лежит хотя бы в одном из трёх прямоугольников.
        """
        # Входной канал
        in_inlet = (self.x_min <= x <= -self.x_min) and (self.y_min <= y <= -self.W_side / 2)
        # Верхний выход
        in_up = (self.x_min <= x <= -self.x_min) and (self.W_side / 2 <= y <= self.y_max)
        # Боковой выход
        in_side = (-self.x_min <= x <= self.x_max) and (-self.W_side / 2 <= y <= self.W_side / 2)
        # Центр тройника
        in_centere = (-self.W_in / 2 <= x <= self.W_in / 2) or (-self.W_side / 2 <= y <= self.W_side / 2)

        return in_inlet or in_up or in_side or in_centere

    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Возвращает (xmin, xmax, ymin, ymax) ограничивающего прямоугольника."""
        return (self.x_min, self.x_max, self.y_min, self.y_max)


class UniformGrid:
    """
    Равномерная прямоугольная сетка внутри заданной геометрии.
    Точки сетки соответствуют центрам ячеек размером cell_size x cell_size.
    """
    def __init__(self, geometry: TGeometry, cell_size: float):
        self.geometry = geometry
        self.cell_size = cell_size
        self._points: Optional[List[Tuple[float, float]]] = None

    @property
    def points(self) -> List[Tuple[float, float]]:
        """Генерирует (при необходимости) и возвращает список координат точек сетки."""
        if self._points is None:
            self._points = self._generate()
        return self._points

    def _generate(self) -> List[Tuple[float, float]]:
        """Построение точек сетки с заданным шагом."""
        xmin, xmax, ymin, ymax = self.geometry.bounding_box()
        dx = self.cell_size
        dy = self.cell_size
        points = []

        # Центры ячеек смещены на половину шага от границ
        x = xmin + dx / 2
        while x <= xmax:
            y = ymin + dy / 2
            while y <= ymax:
                if self.geometry.contains(x, y):
                    points.append((x, y))
                y += dy
            x += dx

        return points

    def visualize(self, show_points: bool = True, show_axes: bool = True,
                  title: str = "Тройник VMFL010. Сеточная модель") -> None:
        """
        Отображает геометрию тройника и точки сетки (если show_points=True).
        Требуется установленный matplotlib.
        """

        xmin, xmax, ymin, ymax = self.geometry.bounding_box()
        fig, ax = plt.subplots(figsize=(8, 8))

        # Нижний канал
        ax.add_patch(Rectangle((-self.geometry.W_in / 2, -(self.geometry.L_in + self.geometry.W_side / 2)),
                                    self.geometry.W_in, self.geometry.L_in,
                                    fill=False, edgecolor='black', linewidth=1))
        # Верхний канал
        ax.add_patch(Rectangle((-self.geometry.W_in / 2, self.geometry.W_side / 2),
                                    self.geometry.W_in, self.geometry.L_up,
                                    fill=False, edgecolor='black', linewidth=1))
        # Боковой канал
        ax.add_patch(Rectangle((self.geometry.W_in / 2, -self.geometry.W_side/2),
                                    self.geometry.L_side, self.geometry.W_side,
                                    fill=False, edgecolor='black', linewidth=1))
        # Центр
        ax.add_patch(Rectangle((-self.geometry.W_in / 2, -self.geometry.W_side/2),
                                    self.geometry.W_in, self.geometry.W_side,
                                    fill=False, edgecolor='black', linewidth=1))

        if show_points and self.points:
            xs, ys = zip(*self.points)
            ax.plot(xs, ys, 'b.', markersize=2, label='Точки сеточной модели')

        ax.set_xlim(xmin-1, xmax+1)
        ax.set_ylim(ymin-1, ymax+1)
        ax.set_aspect('equal')
        ax.set_title(title)
        if show_axes:
            ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
            ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.legend(loc='upper right')
        # ax.grid()
        plt.savefig(f'junction_PINN/data/geom/{self.geometry.name}')
        plt.show()
        


def load_config_from_yaml(filepath: str) -> Tuple[TGeometry, UniformGrid]:
    """
    Загружает конфигурацию из YAML файла.
    Ожидаемая структура:
        geometry:
            L_in: ...
            W_in: ...
            L_up: ...
            W_up: ...
            L_side: ...
            W_side: ...
        grid:
            cell_size: ...
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    geom_data = data['geometry']
    geometry = TGeometry(
        name=geom_data['name'],
        L_in=geom_data['L_in'],
        L_up=geom_data['L_up'],
        L_side=geom_data['L_side'],
        W_in=geom_data['W_in'],
        W_side=geom_data['W_side']
    )
    cell_size = data['grid']['cell_size']
    grid = UniformGrid(geometry, cell_size)
    return geometry, grid


# Пример использования
if __name__ == '__main__':

    geom, grid = load_config_from_yaml('junction_PINN/geom_config.yaml')

    print(f"Точек, всего : {len(grid.points)}")


    grid.visualize()