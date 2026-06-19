# Airfoil PINN UI

PySide6 interface developed on the `UI` branch.

Run from the project root:

```powershell
python -m ui.app
```

Install UI dependencies:

```powershell
python -m pip install -r ui/requirements.txt
```

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
