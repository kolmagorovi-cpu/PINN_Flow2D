"""Read the same normalized coordinates used by the training dataset."""

import numpy as np

from airfoil_pinn.app.paths import resolve_project_path


def read_coordinate_bounds(points_file: str) -> tuple[float, float, float, float]:
    """Return x_min, x_max, y_min, y_max from the selected initial fields."""
    data = np.load(resolve_project_path(points_file), allow_pickle=True)
    if not isinstance(data, np.ndarray) or data.dtype.names is None:
        if hasattr(data, 'close'):
            data.close()
        raise ValueError("Файл точек должен быть структурированным массивом")
    bounds = []
    for field in ('x_norm', 'y_norm'):
        if field not in data.dtype.names:
            raise ValueError(f"В файле точек отсутствует поле {field}")
        # Dataset converts coordinates to float32 before Solver takes extrema.
        values = np.asarray(data[field], dtype=np.float32)
        if not values.size or not np.isfinite(values).all():
            raise ValueError(f"Поле {field} пусто или содержит NaN/Inf")
        bounds.extend((float(values.min()), float(values.max())))
    return tuple(bounds)
