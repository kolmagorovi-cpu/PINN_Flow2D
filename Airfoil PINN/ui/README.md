# Airfoil PINN UI

PySide6 desktop interface for running Airfoil PINN training from a separate
backend process.

Install UI dependencies from the project root:

```powershell
python -m pip install -r requirements_ui.txt
```

Run the UI:

```powershell
python main_ui.py
```

Run UI smoke checks:

```powershell
python -m ui.smoke_tests
```

The Training panel can run `PINN_Learning.py --config ... --experiment_dir ...`
or a quick `Mock backend` subprocess for UI testing. Progress is parsed from
backend stdout lines such as `Epoch 100/1000, Loss: 0.00123`.

The `Geometry` screen is implemented in UI code without modifying backend files.
It can:

- load `.txt`, `.dat`, `.csv`, `.npy` airfoil coordinate files;
- accept drag-and-drop geometry files;
- accept manual coordinates in `x y`, `x,y`, or `x; y` form;
- load from `geom_config.yaml`;
- load from `Geometry/foil_geoms.yaml` when that library exists;
- detect closed and open profiles;
- build a domain using `front`, `rear`, `top`, and `bottom`;
- preview the profile and domain interactively;
- save geometry artifacts to `Geometry/Data` and `Geometry/Metadata`.

Generated files follow the backend naming pattern:

- `Geometry/Metadata/<name>_foil.<format>`
- `Geometry/Data/<name>_domain.<format>`
- `Geometry/Metadata/<name>_metadata.npy`

Training UI features:

- preset/profile selection and custom coordinate file browsing;
- epochs, learning rate, batch size, collocation point count, device selection;
- PDE/BC/data loss weights and FVM residual toggle;
- asynchronous backend launch through `BackendWorker` in `QThread`;
- buffered real-time log streaming and progress-bar updates from stdout parsing;
- non-blocking stop requests for the backend process tree;
- validation that real backend YAML configs contain existing `data.points_file`
  and, when FVM is enabled, `data.mesh_file`;
- JSON save/load for reproducible UI configurations;
- automatic restoration of the latest UI settings with `QSettings`;
- automatic loading of saved PNG/JPG plots from a results folder.
