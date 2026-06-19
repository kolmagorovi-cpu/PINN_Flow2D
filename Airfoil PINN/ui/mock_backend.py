"""Tiny subprocess backend used to test the UI without running a PINN job."""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
import zlib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mock Airfoil PINN backend")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--fail", action="store_true")
    return parser.parse_args()


def save_mock_plots(output_dir: Path, losses: list[float]) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _write_simple_png(plots_dir / "learning_curve_mock.png", "loss")
        _write_simple_png(plots_dir / "velocity_field_mock.png", "velocity")
        _write_simple_png(plots_dir / "pressure_field_mock.png", "pressure")
        (output_dir / "mock_results.txt").write_text(
            "matplotlib is not installed, fallback PNG files were written\n",
            encoding="utf-8",
        )
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(range(1, len(losses) + 1), losses, color="#1264a3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Mock learning curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "learning_curve_mock.png", dpi=150)
    plt.close(fig)

    xs = [i / 40 for i in range(41)]
    ys = [j / 30 - 0.5 for j in range(31)]
    for filename, title, phase in (
        ("velocity_field_mock.png", "Mock velocity field", 0.0),
        ("pressure_field_mock.png", "Mock pressure field", math.pi / 3),
    ):
        grid = [
            [math.sin(2 * math.pi * x + phase) * math.cos(2 * math.pi * y) for x in xs]
            for y in ys
        ]
        fig, ax = plt.subplots(figsize=(6, 4))
        image = ax.imshow(
            grid,
            extent=[min(xs), max(xs), min(ys), max(ys)],
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        ax.plot([0.05, 0.5, 0.95, 0.05], [0.0, 0.08, 0.0, 0.0], color="white", linewidth=1.4)
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(plots_dir / filename, dpi=150)
        plt.close(fig)


def _write_simple_png(path: Path, kind: str, width: int = 320, height: int = 200) -> None:
    def pixel(x: int, y: int) -> tuple[int, int, int]:
        nx = x / max(1, width - 1)
        ny = y / max(1, height - 1)
        if kind == "loss":
            curve = 1.0 - 0.78 * nx
            bright = 245 if abs(ny - curve) < 0.012 else 30
            return 18, min(120 + bright // 2, 255), min(160 + bright // 3, 255)
        if kind == "velocity":
            value = int(127 + 90 * math.sin(8 * nx) * math.cos(6 * ny))
            return 20, max(0, min(255, value + 60)), max(0, min(255, 210 - value // 3))
        value = int(127 + 80 * math.sin(7 * nx + 2 * ny))
        return max(0, min(255, value + 40)), max(0, min(255, 170 - value // 4)), 90

    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend(pixel(x, y))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(name: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    losses: list[float] = []
    loss = 1.0

    print("Mock backend started", flush=True)
    for epoch in range(1, args.epochs + 1):
        loss *= 0.92
        pde = loss * 0.58
        bc = loss * 0.27
        data = loss * 0.15
        losses.append(loss)
        print(
            f"Epoch {epoch}/{args.epochs}, Loss: {loss:.6e}, "
            f"PDE: {pde:.6e}, BC: {bc:.6e}, Data: {data:.6e}",
            flush=True,
        )
        time.sleep(max(args.delay, 0.0))

    if args.fail:
        print("Mock backend requested failure", file=sys.stderr, flush=True)
        return 2

    save_mock_plots(args.output_dir, losses)
    report = (
        f"final_loss: {losses[-1]:.6e}\n"
        f"final_pde_loss: {losses[-1] * 0.58:.6e}\n"
        f"final_bc_loss: {losses[-1] * 0.27:.6e}\n"
        f"best_epoch: {len(losses)}\n"
    )
    (args.output_dir / "final_report.txt").write_text(report, encoding="utf-8")
    print(f"Mock backend finished. Results: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
