"""Диагностический тест FVM-настроек."""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

errors = []

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.training.components.loss_computer import LossComputer

cfg = ConfigManager()
lw = cfg.config['loss_weights']

if 'fvm' not in lw:
    errors.append("FAIL: 'fvm' не найден в loss_weights")
else:
    fvm = lw['fvm']
    if not isinstance(fvm, dict):
        errors.append(f"FAIL: fvm должен быть dict, получено {type(fvm)}")
    elif fvm.get('enabled') is not False:
        errors.append(f"FAIL: fvm.enabled={fvm.get('enabled')}, ожидалось False")
    elif fvm.get('weight') != 1.0:
        errors.append(f"FAIL: fvm.weight={fvm.get('weight')}, ожидалось 1.0")
    else:
        print("OK  ConfigManager._default_config() содержит fvm.enabled=False, fvm.weight=1.0")

# ── 2. ConfigManager save / load ─────────────────────────────────────────────
lw['fvm']['enabled'] = True
lw['fvm']['weight'] = 2.5
cfg.name = '__test_fvm__'
cfg.save()

cfg2 = ConfigManager()
cfg2.load('__test_fvm__')
fvm2 = cfg2.config['loss_weights'].get('fvm', {})
if fvm2.get('enabled') is not True:
    errors.append(f"FAIL: после load fvm.enabled={fvm2.get('enabled')}, ожидалось True")
elif fvm2.get('weight') != 2.5:
    errors.append(f"FAIL: после load fvm.weight={fvm2.get('weight')}, ожидалось 2.5")
else:
    print("OK  ConfigManager save/load сохраняет fvm.enabled и fvm.weight корректно")

os.remove(os.path.join(ConfigManager.CONFIG_DIR, '__test_fvm__.yaml'))

# ── 3. LossComputer — новый формат {enabled: True, weight: X} ────────────────
def make_lw(fvm_val):
    return {
        'pde_weights': {'cont': 1.0, 'mom_x': 1.0, 'mom_y': 1.0, 'mom_r': 1.0, 'mom_z': 1.0},
        'bc_weights': {'inlet': 1.0, 'outlet': 1.0, 'wall': 1.0,
                       'symmetry': 1.0, 'axis_symmetry': 1.0,
                       'symmetry_grad': 1.0, 'axis_symmetry_grad': 1.0},
        'adaptive_bc': False,
        'use_sa_weights': False,
        'sa_apply_to_bc': False,
        'integral_condition': False,
        'integral_weight': 0.0,
        'integral_target': 0.0,
        'fvm': fvm_val,
    }

import torch


class _FakeDataHandler:
    boundary_type_names = ['inlet', 'outlet', 'wall']
    inlet_u_norm = 1.0
    inlet_v_norm = 0.0


def _make_lc(fvm_val):
    cfg_dict = {
        'pde': {'flow_type': '2d'},
        'boundary_conditions': {'type': 'soft'},
        'loss_weights': make_lw(fvm_val),
        'input_params': {'inlet_velocity': False, 'viscosity': False, 'density': False},
    }
    lc = LossComputer(
        config=cfg_dict,
        data_handler=_FakeDataHandler(),
        rho_norm=torch.tensor(1.0),
        mu_norm=torch.tensor(1.0),
        x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
        device=torch.device('cpu'),
        dtype=torch.float32,
    )
    return lc

cases = [
    # (fvm_val, expected_weight, description)
    ({},                                0.0, "старый формат {} -> weight=0"),
    ({'weight': 0.5},                   0.5, "старый формат {weight:0.5} без enabled: weight=0.5"),
    ({'enabled': False, 'weight': 1.0}, 0.0, "новый формат enabled=False: weight=0"),
    ({'enabled': True,  'weight': 2.0}, 2.0, "новый формат enabled=True, weight=2.0"),
    (0.75,                             0.75, "legacy число 0.75: weight=0.75"),
]

for fvm_val, expected, desc in cases:
    try:
        lc = _make_lc(fvm_val)
        if abs(lc.fvm_weight - expected) > 1e-9:
            errors.append(f"FAIL: {desc}: fvm_weight={lc.fvm_weight}, ожидалось {expected}")
        else:
            print(f"OK  LossComputer: {desc}")
    except Exception as e:
        errors.append(f"FAIL: {desc}: исключение {e}")

# ── Итог ─────────────────────────────────────────────────────────────────────
print()
if errors:
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("=== Все проверки пройдены успешно ===")
