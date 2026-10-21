#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI-инспектор CFD-набора: ``python -m pipe_bend_pinn_data --manifest ...``.

Позволяет проверить манифест, разобрать CSV, увидеть сводку по кейсам
(число точек, U_in, Re, диапазон Cp, SDF) и сохранить нормализованный
структурированный массив для обучения.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .handler import CFDataHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipe_bend_pinn_data",
        description="Загрузка и валидация CFD-кейсов для параметрической PINN",
    )
    parser.add_argument("--manifest", required=True, help="путь к YAML/JSON-манифесту")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="не падать на ошибках валидации (только предупреждения)",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="сохранить нормализованные точки в .npy (structured array)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )

    handler = CFDataHandler(args.manifest, strict=not args.no_strict).load()
    print(json.dumps(handler.summary(), indent=2, ensure_ascii=False))

    if args.save:
        path = handler.save_npy(args.save)
        print(f"\nСохранено: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
