#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Фоновый мониторинг обучения Pipe-Bend PINN.

Каждые INTERVAL секунд делает снимок состояния обучения и дописывает строку
в workspace/logs/train_status.txt. Работает как отдельный процесс, не зависит
от IDE.

Снимок: timestamp | epoch | loss | pde_max | gpu_util% | gpu_mem_MiB | alive

Запуск (отдельно от обучения):
    python scripts/monitor_training.py
"""

from __future__ import annotations

import csv
import datetime
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "workspace", "experiments")
STATUS_PATH = os.path.join(PROJECT_ROOT, "workspace", "logs", "train_status.txt")
INTERVAL_SECONDS = 300  # 5 минут

# Порог завершения: обучение считается завершённым, когда достигнута эта эпоха
# (из конфига) или процесс обучения больше не пишет в лог.
TOTAL_EPOCHS = 20000
LBFGS_EPOCH = 16000


def _gpu_state() -> str:
    """Возвращает строку 'util%,mem_MiB' из nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        text = out.stdout.strip()
        if text:
            parts = [p.strip() for p in text.split(",")]
            util = parts[0].replace("%", "").strip()
            return f"{util}%,{parts[1]}"
    except Exception:
        pass
    return "n/a"


def _training_state() -> tuple:
    """Возвращает (epoch, loss, pde_max) из metrics.csv последнего эксперимента."""
    if not os.path.isdir(EXPERIMENTS_DIR):
        return None, None, None
    candidates = [
        os.path.join(EXPERIMENTS_DIR, name)
        for name in sorted(os.listdir(EXPERIMENTS_DIR))
        if name.startswith("pipe_bend_laminar")
    ]
    if not candidates:
        return None, None, None
    latest = candidates[-1]
    csv_path = os.path.join(latest, "logs", "metrics.csv")
    if not os.path.exists(csv_path):
        return None, None, None
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        if len(rows) < 2:
            return None, None, None
        last = rows[-1]
        epoch = last[0] if len(last) > 0 else None
        loss = last[1] if len(last) > 1 else None
        pde_max = last[-1] if len(last) > 1 else None
        return epoch, loss, pde_max
    except Exception:
        return None, None, None


def snapshot() -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    epoch, loss, pde_max = _training_state()
    gpu = _gpu_state()
    try:
        epoch_i = int(float(epoch)) if epoch else 0
        phase = "L-BFGS" if epoch_i >= LBFGS_EPOCH else "Adam"
    except (TypeError, ValueError):
        phase = "?"
    done = "DONE" if (epoch and int(float(epoch)) >= TOTAL_EPOCHS) else "running"
    line = (
        f"{ts} | epoch={epoch or '?'} | loss={loss or '?'} | "
        f"pde_max={pde_max or '?'} | gpu={gpu} | phase={phase} | {done}"
    )
    with open(STATUS_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def main() -> int:
    os.makedirs(os.path.dirname(STATUS_PATH), exist_ok=True)
    print(f"Мониторинг запущен. Снимки каждые {INTERVAL_SECONDS} с -> {STATUS_PATH}")
    while True:
        try:
            snapshot()
        except Exception as exc:  # монитор не должен умирать
            print(f"snapshot error: {exc}", flush=True)
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
