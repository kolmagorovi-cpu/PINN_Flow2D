"""Rebuild the illustrated FVM3 report from local artifacts; never train a model.

Run from the repository root with ``venv/Scripts/python.exe docs/build_fvm3_guide.py``.
Requires the training environment plus Python-Markdown. Outputs stay in docs/.
"""
from __future__ import annotations

import base64
import hashlib
import html
import importlib.metadata
import io
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Polygon
import markdown
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT / "docs"
ASSETS = DOCS / "fvm3_assets"
RUN = ROOT / "workspace/experiments/fvm3_20261021_200705"
BLUE, ORANGE, GREEN, DARK = "#2563a6", "#dc6b30", "#178579", "#20324a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": DARK, "text.color": DARK})


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ASSETS / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, color=BLUE, fontsize=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018",
                               facecolor=color, edgecolor="none", alpha=.10))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, a, b):
    ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": DARK})


def figures(points, mesh, rows):
    coords, tri, foil = mesh["coords"], mesh["triangles"], mesh["airfoil_coords"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), gridspec_kw={"height_ratios": [1, 1]})
    ax = axes[0]
    ax.fill(foil[:, 0], foil[:, 1], color=DARK)
    for x, y, color in [([-1, -1], [-1, 1], BLUE), ([-1, 5], [1, 1], GREEN),
                         ([-1, 5], [-1, -1], GREEN), ([5, 5], [-1, 1], GREEN)]:
        ax.plot(x, y, color=color, lw=3)
    ax.text(-1.08, 0, "inlet\nu*=1, v*=0", ha="right", va="center", color=BLUE)
    ax.text(5.08, 0, "farfield\nu*=1, v*=0", va="center", color=GREEN)
    ax.text(2, 1.10, "farfield: u*=1, v*=0", ha="center", color=GREEN)
    ax.text(2, -1.25, "farfield: u*=1, v*=0", ha="center", color=GREEN)
    ax.annotate("wall: u*=v*=0", xy=(.5, .04), xytext=(1.1, .45),
                arrowprops={"arrowstyle": "->"})
    for y in [-.6, 0, .6]: arrow(ax, (-.8, y), (-.3, y))
    ax.set(xlim=(-1.85, 5.85), ylim=(-1.45, 1.45), xlabel="x/c", ylabel="y/c",
           title="Фактическая область: [-1, 5] × [-1, 1]; справа нет outlet")
    ax.set_aspect("equal")
    ax = axes[1]
    ax.fill(foil[:, 0], foil[:, 1], color=DARK, alpha=.85)
    ax.plot(foil[:, 0], foil[:, 1], color=DARK, lw=1)
    ax.annotate("c = 1 м", xy=(0, -.11), xytext=(1, -.11), ha="center",
                arrowprops={"arrowstyle": "<->"})
    ax.text(.3, .085, "NACA 0012; толщина ≈ 0,12 c; α = 0°", ha="center")
    ax.set(xlim=(-.12, 1.15), ylim=(-.17, .15), xlabel="x/c", ylabel="y/c")
    ax.set_aspect("equal")
    fig.tight_layout(h_pad=2)
    save(fig, "01_geometry.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].triplot(coords[:, 0], coords[:, 1], tri, color=BLUE, lw=.25, alpha=.65)
    axes[0].fill(foil[:, 0], foil[:, 1], color=DARK, zorder=3)
    axes[0].set(xlim=(-.025, .17), ylim=(-.07, .09), xlabel="x/c", ylabel="y/c",
                title="Реальная сетка у передней кромки")
    axes[0].set_aspect("equal")
    d = points["sdf_norm"][points["boundary_type"] == "interior"]
    axes[1].hist(d, bins=np.geomspace(d.min()*.99, d.max()*1.01, 46), color=GREEN)
    axes[1].set(xscale="log", xlabel="SDF/c", ylabel="Число внутренних точек",
                title="Распределение 16 191 collocation-точки")
    axes[1].axvline(.01, color=ORANGE, linestyle="--", label="70,6% точек: SDF/c < 0,01")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "02_mesh.png")

    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis("off")
    for x, w, text in [(.02, .17, "Вход: 3\nx/c, y/c, d(x,y)/c"),
                        (.26, .19, "Linear 3 → 258\nsin(30z)\n1 032 параметра"),
                        (.52, .19, "Linear 258 → 258\nsin(30z)\n66 822 параметра"),
                        (.78, .19, "Linear 258 → 3\nu*, v*, Cp\n777 параметров")]:
        box(ax, x, .65, w, .27, text)
    for a,b in [(.19,.26),(.45,.52),(.71,.78)]: arrow(ax,(a,.785),(b,.785))
    box(ax,.03,.13,.28,.29,"Autograd по x*, y*\n1-е и 2-е производные\nPDE: масса и импульс",GREEN)
    box(ax,.36,.13,.28,.29,"Маски граничных типов\nBC + фиксация давления\nГлобальный интеграл",ORANGE)
    box(ax,.69,.13,.28,.29,"Отдельный проход:\nвсе 17 325 узлов FVM\nбаланс массы в каждом КО",GREEN)
    for x in [.17,.5,.83]: arrow(ax,(.87,.64),(x,.43))
    ax.text(.5,.025,"Суммарная потеря → backward → Adam → обновление 68 631 параметра",
            ha="center",fontsize=12)
    save(fig,"03_architecture.png")

    # An actual median-dual control volume around a node in the far field.
    i = int(np.argmin(np.sum((coords-np.array([2.4,.45]))**2,axis=1)))
    adj = tri[np.any(tri==i,axis=1)]
    neighbors=np.unique(adj); neighbors=neighbors[neighbors!=i]
    verts=np.vstack([coords[adj].mean(axis=1), (coords[neighbors]+coords[i])/2])
    angles=np.arctan2(verts[:,1]-coords[i,1],verts[:,0]-coords[i,0]); verts=verts[np.argsort(angles)]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    ax=axes[0]
    for t in adj:
        poly=coords[np.r_[t,t[0]]]; ax.plot(poly[:,0],poly[:,1],color="#b0bdc9",lw=1)
    ax.add_patch(Polygon(verts,facecolor=GREEN,alpha=.2,edgecolor=GREEN,lw=2))
    ax.scatter(*coords[i],color=DARK,zorder=5);ax.text(*coords[i],"  i",fontsize=12)
    for j in neighbors: ax.scatter(*coords[j],color=BLUE,s=12)
    ax.set_aspect("equal");ax.set(xlabel="x/c",ylabel="y/c",title="Контрольный объём вокруг реального узла")
    ax=axes[1];ax.axis("off")
    box(ax,.04,.70,.92,.23,"На внутренней грани: u_f = (u_i + u_j)/2")
    box(ax,.04,.39,.92,.23,"Поток Φ_f = u_f · S_ij\nВ узел i: +Φ_f; в узел j: −Φ_f",GREEN)
    box(ax,.04,.08,.92,.23,"R_i = ΣΦ_f / V_i\nL_FVM = Σ(V_i R_i²) / ΣV_i",ORANGE)
    arrow(ax,(.5,.69),(.5,.63));arrow(ax,(.5,.38),(.5,.32))
    fig.tight_layout();save(fig,"04_fvm.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs=[r["epoch"] for r in rows]
    for key,label,color in [("loss","Total (журнал)",DARK),("fvm_weighted_loss","25 × FVM MSE",BLUE),
                             ("integral_loss","Интегральный штраф",ORANGE)]:
        axes[0].semilogy(epochs,[r[key] for r in rows],label=label,color=color)
    axes[0].set(xlabel="Эпоха",ylabel="Потеря",title="Сохранённый запуск: 21.10.2026, 20:07:05")
    for key,color in [("wall",ORANGE),("inlet",BLUE),("farfield",GREEN)]:
        axes[1].semilogy(epochs,[r["boundary_mse"][key] for r in rows],label=key,color=color)
    axes[1].set(xlabel="Эпоха",ylabel="BC MSE",title="Граничные ошибки: завершённые эпохи 1–59")
    for ax in axes: ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();save(fig,"05_training.png")

    # Export unchanged original plot; its provenance is recorded in the guide.
    (ASSETS/"06_saved_u.png").write_bytes((RUN/"plots/field_u_epoch_60.png").read_bytes())


def audit(points, mesh, rows):
    import torch
    from airfoil_pinn.config.manager import ConfigManager
    from airfoil_pinn.data.handler import DataHandler
    from airfoil_pinn.physics.fvm import FVMResiduals
    from airfoil_pinn.training.components.loss_computer import LossComputer
    cfg=ConfigManager(); cfg.load_from_file(str(ROOT/"workspace/configs/fvm3.yaml"))
    dh=DataHandler(points_file=cfg.config["data"]["points_file"],batch_size=5000,
                   mesh_file=cfg.config["data"]["mesh_file"])
    fvm=FVMResiduals.from_mesh_data(dh.mesh_data).to("cpu",dtype=torch.float64)
    fvm.nondimensionalize_geometry(dh.chord)
    lc=LossComputer(cfg.config,dh,torch.tensor(1.),torch.tensor(1./dh.Re),
                    -1.,5.,-1.,1.,torch.device("cpu"),torch.float64)
    uniform=torch.zeros((fvm.N,2),dtype=torch.float64);uniform[:,0]=1
    integral=lc.compute_mesh_integral_loss(uniform,fvm.bnd_node,fvm.bnd_S,fvm.bnd_type)
    model_cfg=cfg.config["architecture"]
    from airfoil_pinn.training.models import PINNIncompressible
    model=PINNIncompressible(3,model_cfg["hidden_layers"],activation="siren",
                            output_bias=model_cfg["output_bias"],coordinate_scaling=model_cfg["coordinate_scaling"])
    files=[ROOT/"workspace/configs/fvm3.yaml",ROOT/"workspace/initial/fields/initial.npy",
           ROOT/"workspace/initial/fields/initial_metadata.npy",ROOT/"workspace/sampling/mesh/mesh2d.npz",
           ROOT/"workspace/sampling/collocation/collocation_points.npy",
           ROOT/"workspace/sampling/collocation/collocation_points_metadata.npy"]
    files += list((ROOT/"workspace/geometry/metadata").glob("NACA*"))
    files += list((ROOT/"workspace/geometry/domains").glob("NACA*"))
    files += list((ROOT/"airfoil_pinn").rglob("*.py"))
    versions={name:importlib.metadata.version(name) for name in
              ["torch","numpy","scipy","matplotlib","PyYAML","shapely","typing_extensions","tqdm","psutil","Markdown"]}
    ckpt=torch.load(RUN/"checkpoints/pinn_interrupted.pt",map_location="cpu",weights_only=False)
    result={"report_date":"2026-10-23", "python":sys.version,"packages":versions,
            "cuda_available":torch.cuda.is_available(),
            "gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "points":len(points), "initial_dtype":points.dtype.descr,
            "boundary_counts":dict(zip(*[x.tolist() for x in np.unique(points["boundary_type"],return_counts=True)])),
            "mesh":fvm.summary(),"model_parameters":sum(p.numel() for p in model.parameters()),
            "uniform_velocity_integral_loss":float(integral),
            "geometry_closes":fvm.check_geometry_conservativity(),
            "checkpoint_epoch":ckpt.get("epoch"),"completed_log_epochs":len(rows),
            "first_epoch":rows[0],"last_epoch":rows[-1],
            "sha256":{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (ASSETS/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (ASSETS/"requirements-observed.txt").write_text(
        "# Observed environment, 2026-10-23; CUDA wheel requires the matching PyTorch index.\n"+
        "\n".join(f"{n}=={v}" for n,v in versions.items())+"\n",encoding="utf-8")
    import yaml
    (ASSETS/"fvm3_effective.yaml").write_text(yaml.safe_dump(cfg.config,allow_unicode=True,sort_keys=False),encoding="utf-8")
    print(json.dumps({k:result[k] for k in ["packages","points","model_parameters","uniform_velocity_integral_loss","geometry_closes","checkpoint_epoch"]}))


def render():
    source=(DOCS/"FVM3_EXPERIMENT.md").read_text(encoding="utf-8")
    equations=[]
    def equation(match):
        expr=match.group(1).strip()
        stream=io.BytesIO()
        mathtext.math_to_image("$"+expr+"$",stream,format="svg",dpi=180,
                              prop=FontProperties(size=14))
        uri="data:image/svg+xml;base64,"+base64.b64encode(stream.getvalue()).decode()
        equations.append('<div class="equation"><img alt="'+html.escape(expr,quote=True)+'" src="'+uri+'"></div>')
        return "\n\nEQUATIONTOKEN"+str(len(equations)-1)+"END\n\n"
    source=re.sub(r"\$\$(.*?)\$\$",equation,source,flags=re.S)
    body=markdown.markdown(source,extensions=["tables","fenced_code","toc"],extension_configs={"toc":{"toc_depth":"2-3"}})
    for i,eq in enumerate(equations):body=body.replace(f"<p>EQUATIONTOKEN{i}END</p>",eq)
    def embed(match):
        name=match.group(1);path=DOCS/name
        if not path.is_file():raise FileNotFoundError(path)
        mime="image/png" if path.suffix==".png" else "image/svg+xml"
        return 'src="data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()+'"'
    body=re.sub(r'src="(fvm3_assets/[^\"]+)"',embed,body)
    css="""body{font:17px/1.65 'Segoe UI',Arial,sans-serif;color:#20324a;background:#edf2f7;margin:0}
    main{max-width:1060px;margin:auto;background:white;padding:50px 64px}h1{font-size:36px;line-height:1.2}
    h2{font-size:26px;margin-top:2.1em;border-bottom:2px solid #dce6ef;padding-bottom:9px}h3{font-size:21px}
    a{color:#1f60a0}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}
    th,td{padding:9px 12px;border:1px solid #dce6ef;text-align:left;vertical-align:top}th{background:#eef5fa}
    code{font-size:.88em;overflow-wrap:anywhere}pre{padding:18px;background:#f1f5f9;overflow:auto;line-height:1.4}
    blockquote{margin:20px 0;padding:12px 22px;border-left:4px solid #dc6b30;background:#fff5ee}
    .equation{text-align:center;margin:26px 0;overflow:auto}.equation img{max-height:85px}
    .toc{padding:18px 25px;background:#f1f5f9}.toc ul{margin:6px 0}
    @media print{body{background:white;font-size:11pt}main{padding:0;max-width:none}h2,h3{break-after:avoid}
    img,table,.equation{break-inside:avoid}pre{white-space:pre-wrap}a{color:inherit}}
    @media(max-width:700px){main{padding:24px}table{display:block;overflow:auto}}
    """
    (DOCS/"FVM3_EXPERIMENT.html").write_text('<!doctype html><html lang="ru"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>FVM3 — инструкция по эксперименту</title>'
        '<style>'+css+'</style><main>'+body+'</main></html>',encoding="utf-8")


def main():
    ASSETS.mkdir(exist_ok=True)
    points=np.load(ROOT/"workspace/initial/fields/initial.npy",allow_pickle=True)
    with np.load(ROOT/"workspace/sampling/mesh/mesh2d.npz",allow_pickle=True) as data:
        mesh={k:data[k] for k in data.files}
    rows=[json.loads(line) for line in (RUN/"logs/epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    audit(points,mesh,rows)
    figures(points,mesh,rows)
    render()
    print("Wrote docs/FVM3_EXPERIMENT.html and fvm3_assets/")


if __name__=="__main__":
    main()
