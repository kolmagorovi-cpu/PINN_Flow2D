#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Построение схемы нейронной сети PINN.

Модуль отделяет графическую диагностическую функцию от консольного меню
обучения. Меню передает сюда конфигурацию, а модуль отвечает только за
формирование PNG-файла со схемой входов, скрытых слоев и выходов сети.
"""

import os
from typing import Any

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from airfoil_pinn.app.paths import PICTURES_DIR

__all__ = ["PICTURES_DIR", "visualize_network"]


def visualize_network(config: dict[str, Any], filename: str) -> None:
    """
    Строит схему нейронной сети и сохраняет ее в PNG.

    Args:
        config: полная конфигурация проекта.
        filename: имя файла внутри директории ``workspace/pictures``.
    """
    input_params = config.get("input_params", {})
    arch = config.get("architecture", {})
    hidden_layers = arch.get("hidden_layers", [50, 50, 50, 50])
    activation = arch.get("activation", "tanh")
    output_dim = arch.get("output_dim", 3)

    # Подписи входов должны соответствовать флагам из input_params.
    input_labels = []
    if input_params.get("coordinates", True):
        input_labels.extend(["x", "y"])
    if input_params.get("sdf", True):
        input_labels.append("sdf")
    if input_params.get("boundary_types", True):
        input_labels.append("boundary (one-hot)")
    if input_params.get("inlet_velocity", False):
        input_labels.extend(["u_in", "v_in"])
    if input_params.get("viscosity", False):
        input_labels.append("mu")
    if input_params.get("density", False):
        input_labels.append("rho")

    input_dim = len(input_labels)

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(hidden_layers) * 2), 6))
    ax.set_xlim(0, len(hidden_layers) + 2.5)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_aspect("equal")

    color_input = "#FFDDC1"
    color_hidden = "#C1E1FF"
    color_output = "#D1FFD1"

    box_width = 0.8
    max_neurons = max(hidden_layers) if hidden_layers else 1
    scale = 8.0 / max(max_neurons, 1)

    # Входной слой показывает количество фактических признаков модели.
    x_in = 0.5
    height_in = input_dim * scale
    rect_in = patches.Rectangle(
        (x_in - box_width / 2, 1),
        box_width,
        height_in,
        linewidth=2,
        edgecolor="black",
        facecolor=color_input,
    )
    ax.add_patch(rect_in)
    ax.text(
        x_in,
        1 + height_in / 2,
        f"Input\n{input_dim} neurons",
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    y_text = 1 + height_in + 0.2
    ax.text(
        x_in,
        y_text,
        "\n".join(input_labels[:5]),
        ha="center",
        va="bottom",
        fontsize=7,
    )
    if len(input_labels) > 5:
        ax.text(x_in, y_text - 0.3, "...", ha="center", va="bottom", fontsize=7)

    # Каждый скрытый слой рисуется отдельным блоком с числом нейронов.
    for i, neurons in enumerate(hidden_layers):
        x = 1.5 + i
        height = neurons * scale
        rect = patches.Rectangle(
            (x - box_width / 2, 1),
            box_width,
            height,
            linewidth=2,
            edgecolor="black",
            facecolor=color_hidden,
        )
        ax.add_patch(rect)
        ax.text(
            x,
            1 + height / 2,
            f"Layer {i + 1}\n{neurons}\n{activation}",
            ha="center",
            va="center",
            fontsize=9,
        )

    # Выходной слой зависит от режима: несжимаемый поток или добавленная плотность.
    x_out = 1.5 + len(hidden_layers)
    height_out = output_dim * scale
    rect_out = patches.Rectangle(
        (x_out - box_width / 2, 1),
        box_width,
        height_out,
        linewidth=2,
        edgecolor="black",
        facecolor=color_output,
    )
    ax.add_patch(rect_out)
    outputs = ["u", "v", "p"] if output_dim == 3 else ["u", "v", "p", "rho"]
    ax.text(
        x_out,
        1 + height_out / 2,
        "Output\n" + ", ".join(outputs),
        ha="center",
        va="center",
        fontsize=9,
    )

    plt.title(f"Схема сети: {config.get('project_name', 'unnamed')}", fontsize=14)

    os.makedirs(PICTURES_DIR, exist_ok=True)
    path = os.path.join(PICTURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Схема сети сохранена: {path}")
