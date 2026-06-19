"""Matplotlib image viewer used by the Results panel."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class ResultImageCanvas(FigureCanvasQTAgg):
    """Display saved PNG/JPEG artifacts inside the Qt application."""

    def __init__(self, parent=None) -> None:
        self.figure = Figure(figsize=(4.0, 3.0), dpi=100)
        self.axes = self.figure.add_subplot(111)
        self.figure.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
        self._image_cache: OrderedDict[tuple[str, int, int], Any] = OrderedDict()
        self._current_key: tuple[str, int, int] | None = None
        super().__init__(self.figure)
        self.setParent(parent)
        self.setMinimumHeight(260)
        self.show_message("No result image selected")

    def show_message(self, text: str) -> None:
        self._current_key = None
        self.axes.clear()
        self.axes.axis("off")
        self.axes.text(0.5, 0.5, text, ha="center", va="center", wrap=True)
        self.draw_idle()

    def load_image(self, path: str | Path) -> None:
        image_path = Path(path)
        if not image_path.exists():
            self.show_message(f"Missing file:\n{image_path}")
            return

        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            self.show_message(f"Unsupported image:\n{image_path.name}")
            return

        stat = image_path.stat()
        key = (str(image_path.resolve()), stat.st_mtime_ns, stat.st_size)
        if self._current_key == key:
            return

        try:
            data = self._image_cache.pop(key)
            self._image_cache[key] = data
        except KeyError:
            try:
                import matplotlib.image as mpimg

                data = mpimg.imread(image_path)
            except Exception as exc:
                self.show_message(f"Could not load image:\n{exc}")
                return
            self._image_cache[key] = data
            while len(self._image_cache) > 8:
                self._image_cache.popitem(last=False)
        except Exception as exc:
            self.show_message(f"Could not load image:\n{exc}")
            return

        self._current_key = key
        self.axes.clear()
        self.axes.imshow(data)
        self.axes.set_title(image_path.name, fontsize=9)
        self.axes.axis("off")
        self.draw_idle()
