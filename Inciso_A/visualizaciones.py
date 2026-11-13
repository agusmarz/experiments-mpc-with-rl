"""Figuras del eje A: cómo funciona V_psi y cuánto rinde.

    python Inciso_A/visualizaciones.py --modelo Inciso_A/modelos/V_psi_N5_s0.pt

Genera en Inciso_A/figuras/:
    1_aprendizaje.png      curvas del entrenamiento (historial guardado en el .pt)
    2_valor.png            x'Px, V_psi y la corrección exp(phi) en el plano (theta, x_d)
    3_bellman.png          V_psi contra su target (T^N V_psi) y residuo de Bellman en el plano
    4_regiones.png         región de recuperación por costo terminal y horizonte
    5_horizonte.png        estados recuperados en función de N
    6_trayectorias.png     un estado que V_psi recupera y x'Px no
Los barridos en lazo cerrado se cachean en figuras/barrido.npz (borrarlo para recalcular).
"""

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, TwoSlopeNorm  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "simulador"))
sys.path.insert(0, str(AQUI))
from pendulo_env import PenduloEnv, ParametrosPlanta  # noqa: E402
from mpc_no_lineal import MPCNoLineal, costo_terminal_cuadratico, riccati  # noqa: E402
from valor_aprendido import cargar, targets_bellman  # noqa: E402

# Estilo (paleta de referencia validada; 3 series categóricas)
SUPERFICIE, TINTA, TINTA2, APAGADO, GRILLA, EJE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"x'Px": "#2a78d6", "V_psi": "#eb6834", "sin V_f": "#1baf7a"}
ETIQUETA = {"x'Px": r"$x^\top P x$", "V_psi": r"$V_\psi$", "sin V_f": r"sin $V_f$"}
AZULES = LinearSegmentedColormap.from_list("azules", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
DIVERGENTE = LinearSegmentedColormap.from_list("div", ["#256abf", "#86b6ef", "#f0efec", "#f19f9e", "#c13434"])
plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "font.family": "sans-serif", "font.size": 10, "text.color": TINTA,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA2, "axes.titlesize": 11, "axes.titleweight": "semibold",
    "axes.titlecolor": TINTA, "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRILLA, "grid.linewidth": 0.6,
    "xtick.color": APAGADO, "ytick.color": APAGADO, "xtick.labelcolor": TINTA2, "ytick.labelcolor": TINTA2,
    "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 160,
})

THETAS = np.linspace(-0.7, 0.7, 15)
VELOCIDADES = np.linspace(-1.5, 1.5, 15)
HORIZONTES = (1, 2, 3, 5)
TIPOS = ("sin V_f", "x'Px", "V_psi")
PASOS = 250


# ------------------------------------------------------------------ barrido paralelo

_CACHE = {}


def _mpc(N, tipo, ruta_modelo):
    clave = (N, tipo)
    if clave not in _CACHE:
        p = ParametrosPlanta()
        if tipo == "sin V_f":
            vf = None
        elif tipo == "x'Px":
            vf = costo_terminal_cuadratico(riccati(p)[0])
        else:
            vf = cargar(ruta_modelo)[0].a_casadi()
        _CACHE[clave] = (MPCNoLineal(p, N=N, costo_terminal=vf), PenduloEnv(p))
    return _CACHE[clave]


def lazo_cerrado(mpc, env, estado0, pasos=PASOS, guardar=False):
    estado, _ = env.reset(options={"estado": estado0})
    mpc.reiniciar()
    costo, tray = 0.0, [(estado.copy(), 0.0)]
    for _ in range(pasos):
        F = mpc.solve(estado)
        estado, _, terminado, _, info = env.step([F])
        costo += info["costo"]
        if guardar:
            tray[-1] = (tray[-1][0], F)
            tray.append((estado.copy(), np.nan))
        if terminado:
            return False, costo, float(np.mean(mpc.tiempos_solve)), tray
    ok = abs(estado[2]) < 0.05 and abs(estado[0]) < 0.1
    return bool(ok), costo, float(np.mean(mpc.tiempos_solve)), tray


def _tareas():
    return [(N, tipo, i, j) for N in HORIZONTES for tipo in TIPOS
            for i in range(len(VELOCIDADES)) for j in range(len(THETAS))]


def correr_porcion(k, n, ruta_modelo, salida):
    """Proceso hijo: corre las tareas k, k+n, k+2n... y las guarda en un .npy."""
    torch.set_num_threads(1)
    filas = []
    for N, tipo, i, j in _tareas()[k::n]:
        mpc, env = _mpc(N, tipo, ruta_modelo)
        ok, costo, t, _ = lazo_cerrado(mpc, env, [0.0, VELOCIDADES[i], THETAS[j], 0.0])
        filas.append((HORIZONTES.index(N), TIPOS.index(tipo), i, j, ok, costo, t))
    np.save(salida, np.array(filas, dtype=float))


def barrido(ruta_modelo, cache, procesos=6):
    """Barrido en lazo cerrado repartido en subprocesos independientes
    (ProcessPoolExecutor falla en Windows con WinError 6 en este entorno).
    Con muchos procesos (torch + CasADi en cada uno) Windows se queda sin archivo
    de paginación; 6 anda, y cada porción que falle se reintenta una vez sola."""
    import shutil
    import subprocess
    if cache.exists():
        d = np.load(cache)
        return d["ok"], d["costo"], d["t"]
    forma = (len(HORIZONTES), len(TIPOS), len(VELOCIDADES), len(THETAS))
    ok, costo, t = np.zeros(forma, bool), np.full(forma, np.nan), np.full(forma, np.nan)
    t0 = time.perf_counter()
    # Carpeta de intercambio dentro del repo: el Python de la Microsoft Store
    # virtualiza %TEMP% y cada proceso vería una carpeta distinta.
    tmp = cache.parent / "_porciones"
    tmp.mkdir(exist_ok=True)
    venv = AQUI.parent / ".venv" / "Scripts" / "python.exe"
    python = str(venv) if venv.exists() else sys.executable
    if True:
        def lanzar(k):
            return subprocess.Popen([python, __file__, "--modelo", str(ruta_modelo),
                                     "--porcion", str(k), str(procesos), str(tmp / f"p{k}.npy")])
        hijos = [lanzar(k) for k in range(procesos)]
        fallidas = [k for k, h in enumerate(hijos) if h.wait() != 0]
        for k in fallidas:
            print(f"  reintentando porción {k}", flush=True)
            if lanzar(k).wait() != 0:
                raise RuntimeError(f"la porción {k} del barrido falló dos veces")
        for k in range(procesos):
            for a, b, i, j, o, c, ts in np.load(Path(tmp) / f"p{k}.npy"):
                a, b, i, j = int(a), int(b), int(i), int(j)
                ok[a, b, i, j], costo[a, b, i, j], t[a, b, i, j] = bool(o), c, ts
    shutil.rmtree(tmp)
    print(f"  barrido: {ok.size} episodios en {time.perf_counter() - t0:.0f}s", flush=True)
    np.savez(cache, ok=ok, costo=costo, t=t)
    return ok, costo, t


# ------------------------------------------------------------------ figuras

def fig_aprendizaje(hist, cfg, salida):
    it = [h["iteracion"] for h in hist]
    series = [("mse_log", "Error del ajuste (MSE en log V)", True),
              ("cambio_rel_targets", "Cambio relativo de los targets", True),
              ("frac_saturados", "Fracción de targets en V_max", False)]
    fig, ejes = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, (k, titulo, log) in zip(ejes, series):
        y = np.array([h[k] for h in hist], float)
        ax.plot(it, y, color=COLOR["V_psi"], marker="o", markersize=3)
        ax.set_title(titulo)
        ax.set_xlabel("iteración j")
        if log:
            ax.set_yscale("log")
    ejes[2].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    total = hist[-1]["t_total_s"]
    solves = sum(h["solves_mpc"] for h in hist)
    fig.suptitle(f"Entrenamiento de V_psi  ·  N = {cfg.N_bellman}, {cfg.n_estados} estados, semilla {cfg.semilla}  ·  "
                 f"{solves} solves de MPC en {total / 60:.1f} min",
                 x=0.01, ha="left", fontsize=10, color=TINTA2)
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


def plano(n=81):
    th, v = np.meshgrid(np.linspace(-0.8, 0.8, n), np.linspace(-2.0, 2.0, n))
    X = np.stack([np.zeros_like(th), v, th, np.zeros_like(th)], -1).reshape(-1, 4)
    return th, v, X


def fig_valor(red, salida):
    th, v, X = plano()
    with torch.no_grad():
        x = torch.as_tensor(X)
        cuad = red.cuadratico(x).numpy().reshape(th.shape)
        Vpsi = red(x).numpy().reshape(th.shape)
        corr = torch.exp(red.residuo(x)).numpy().reshape(th.shape)

    fig, ejes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    norm = LogNorm(vmin=max(min(cuad.min(), Vpsi.min()), 1.0), vmax=max(cuad.max(), Vpsi.max()))
    for ax, Z, titulo in [(ejes[0], cuad, r"Parte fija  $x^\top P x$"), (ejes[1], Vpsi, r"$V_\psi(x)$ aprendida")]:
        m = ax.pcolormesh(th, v, Z, cmap=AZULES, norm=norm, shading="auto", rasterized=True)
        ax.contour(th, v, Z, levels=[1e2, 1e3, 1e4], colors=SUPERFICIE, linewidths=0.8)
        ax.set_title(titulo)
    fig.colorbar(m, ax=ejes[:2], label="costo a futuro (escala log)", fraction=0.03, pad=0.02)

    lim = float(np.max(np.abs(np.log2(corr))))
    m2 = ejes[2].pcolormesh(th, v, corr, cmap=DIVERGENTE, shading="auto", rasterized=True,
                            norm=LogNorm(vmin=2 ** -lim, vmax=2 ** lim))
    ejes[2].set_title(r"Corrección  $e^{\varphi_\theta(x)} = V_\psi / x^\top P x$")
    cb = fig.colorbar(m2, ax=ejes[2], fraction=0.05, pad=0.02)
    marcas = [2.0 ** k for k in range(-int(np.floor(lim)), int(np.floor(lim)) + 1)]
    cb.set_ticks(marcas, labels=[f"×{m:g}" if m >= 1 else f"×1/{1 / m:g}" for m in marcas])
    cb.minorticks_off()
    cb.set_label("factor sobre x'Px")
    for ax in ejes:
        ax.set_xlabel(r"$\theta$ [rad]")
        ax.grid(False)
        ax.add_patch(plt.Rectangle((-0.7, -1.5), 1.4, 3.0, fill=False, ec=TINTA2, lw=0.8, ls="--"))
    ejes[0].set_ylabel(r"$\dot x_c$ [m/s]")
    fig.suptitle(r"Corte del espacio de estados en $x_c = 0$, $\dot\theta = 0$  ·  recuadro punteado: grilla de la región de recuperación",
                 x=0.01, y=1.04, ha="left", fontsize=10, color=TINTA2)
    fig.savefig(salida, bbox_inches="tight")
    plt.close(fig)


def fig_bellman(red, cfg, salida):
    """V_psi debería ser punto fijo: V_psi(x) ≈ (T^N V_psi)(x)."""
    p = ParametrosPlanta()
    mpc = MPCNoLineal(p, N=cfg.N_bellman, costo_terminal=red.a_casadi())
    rng = np.random.default_rng(1234)  # estados NUEVOS, distintos de los de entrenamiento
    X = rng.uniform(-cfg.caja, cfg.caja, size=(600, 4))
    y, _ = targets_bellman(mpc, X, cfg.V_max)
    pred = red.valor_numpy(X)
    sat = y >= cfg.V_max

    th, v, Xp = plano(41)
    yp, _ = targets_bellman(mpc, Xp, cfg.V_max)
    vp = red.valor_numpy(Xp)
    invalido = (yp >= cfg.V_max) | (yp < 1e-6) | (vp < 1e-6)
    rp = np.full(len(yp), np.nan)
    rp[~invalido] = np.log(vp[~invalido]) - np.log(yp[~invalido])
    rp = rp.reshape(th.shape)

    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = ejes[0]
    lo, hi = 1.0, cfg.V_max * 3
    ax.plot([lo, hi], [lo, hi], color=TINTA2, lw=1, ls="--", zorder=1)
    ax.scatter(y[~sat], pred[~sat], s=12, color=COLOR["V_psi"], alpha=0.6, lw=0, zorder=2, label="estado recuperable")
    ax.scatter(y[sat], pred[sat], s=12, color=APAGADO, alpha=0.6, lw=0, zorder=2, label=r"target saturado en $V_{max}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"target  $(\mathcal{T}^N V_\psi)(x)$  = costo óptimo del MPC")
    ax.set_ylabel(r"$V_\psi(x)$")
    err = np.median(np.abs(np.log(pred[~sat] / y[~sat])))
    ax.set_title("¿V_psi es punto fijo de Bellman?")
    ax.text(0.03, 0.97, f"600 estados nuevos (no usados al entrenar)\nerror relativo mediano {100 * (np.exp(err) - 1):.1f} %\npunteada: V_psi = T^N V_psi",
            transform=ax.transAxes, va="top", fontsize=9, color=TINTA2)
    ax.legend(loc="lower right")

    ax = ejes[1]
    lim = np.nanmax(np.abs(rp))
    m = ax.pcolormesh(th, v, rp, cmap=DIVERGENTE, norm=TwoSlopeNorm(0, -lim, lim), shading="auto", rasterized=True)
    ax.set_facecolor("#e1e0d9")
    fig.colorbar(m, ax=ax, label=r"$\log V_\psi - \log \mathcal{T}^N V_\psi$", fraction=0.05, pad=0.02)
    ax.set_title(r"Residuo de Bellman, corte $x_c = 0$, $\dot\theta = 0$")
    ax.text(0.98, 0.02, "gris: target saturado", transform=ax.transAxes, ha="right", fontsize=8.5, color=TINTA2)
    ax.set_xlabel(r"$\theta$ [rad]"); ax.set_ylabel(r"$\dot x_c$ [m/s]"); ax.grid(False)
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


def fig_regiones(ok, salida):
    fig, ejes = plt.subplots(len(HORIZONTES), len(TIPOS) + 1, figsize=(11.5, 2.6 * len(HORIZONTES)),
                             sharex=True, sharey=True)
    ext = [THETAS[0] - 0.05, THETAS[-1] + 0.05, VELOCIDADES[0] - 0.107, VELOCIDADES[-1] + 0.107]
    for a, N in enumerate(HORIZONTES):
        for b, tipo in enumerate(TIPOS):
            ax = ejes[a, b]
            ax.imshow(ok[a, b], origin="lower", extent=ext, aspect="auto",
                      cmap=ListedColormap(["#f0efec", COLOR[tipo]]), vmin=0, vmax=1, interpolation="nearest")
            ax.text(0.97, 0.05, f"{ok[a, b].sum()}/{ok[a, b].size}", transform=ax.transAxes, ha="right",
                    fontsize=9, color=TINTA, bbox=dict(fc=SUPERFICIE, ec="none", pad=1.5))
            if a == 0:
                ax.set_title(ETIQUETA[tipo])
        # diferencia V_psi vs x'Px
        ax = ejes[a, -1]
        d = ok[a, 2].astype(int) - ok[a, 1].astype(int)
        ax.imshow(d, origin="lower", extent=ext, aspect="auto", interpolation="nearest",
                  cmap=ListedColormap([COLOR["x'Px"], "#f0efec", COLOR["V_psi"]]), vmin=-1, vmax=1)
        ax.text(0.97, 0.05, f"+{(d > 0).sum()} / −{(d < 0).sum()}", transform=ax.transAxes, ha="right",
                fontsize=9, color=TINTA, bbox=dict(fc=SUPERFICIE, ec="none", pad=1.5))
        if a == 0:
            ax.set_title(r"$V_\psi$ − $x^\top P x$")
        ejes[a, 0].set_ylabel(f"N = {N}\n" + r"$\dot x_{c,0}$ [m/s]")
    for ax in ejes.flat:
        ax.grid(False)
    for ax in ejes[-1]:
        ax.set_xlabel(r"$\theta_0$ [rad]")
    fig.suptitle("Región de recuperación (coloreado = recupera en 5 s)  ·  última columna: naranja = solo V_psi, "
                 "azul = solo x'Px", x=0.01, ha="left", fontsize=10, color=TINTA2)
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


def fig_horizonte(ok, t, salida):
    fig, ejes = plt.subplots(1, 2, figsize=(10, 3.6))
    total = ok.shape[2] * ok.shape[3]
    for b, tipo in enumerate(TIPOS):
        frac = ok[:, b].sum(axis=(1, 2)) / total
        ejes[0].plot(HORIZONTES, frac, marker="o", markersize=5, color=COLOR[tipo], label=ETIQUETA[tipo])
        ejes[0].annotate(ETIQUETA[tipo], (HORIZONTES[-1], frac[-1]), xytext=(6, 0), textcoords="offset points",
                         va="center", color=TINTA2, fontsize=9)
        ms = 1e3 * np.nanmean(t[:, b], axis=(1, 2))
        ejes[1].plot(HORIZONTES, ms, marker="o", markersize=5, color=COLOR[tipo], label=ETIQUETA[tipo])
        desfase = {"sin V_f": -7, "x'Px": 7, "V_psi": 0}[tipo]  # x'Px y sin V_f terminan casi juntos
        ejes[1].annotate(ETIQUETA[tipo], (HORIZONTES[-1], ms[-1]), xytext=(6, desfase), textcoords="offset points",
                         va="center", color=TINTA2, fontsize=9)
    ejes[0].set_title("Estados recuperados vs horizonte")
    ejes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ejes[0].set_ylim(0, None)
    ejes[1].set_title("Tiempo medio de solve")
    ejes[1].set_ylabel("ms"); ejes[1].set_ylim(0, None)
    ejes[1].axhline(20, color=TINTA2, lw=1, ls="--")
    ejes[1].text(HORIZONTES[0], 20, " período de control 20 ms", va="bottom", color=TINTA2, fontsize=8.5)
    for ax in ejes:
        ax.set_xlabel("horizonte N del MPC")
        ax.set_xticks(HORIZONTES)
    for ax in ejes:
        ax.set_xlim(HORIZONTES[0] - 0.3, HORIZONTES[-1] + 1.2)
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


def fig_trayectorias(ok, ruta_modelo, salida):
    # Elijo, con el menor N donde haya, el estado más lejano que V_psi recupera y x'Px no
    elegido = None
    for a, N in enumerate(HORIZONTES):
        idx = np.argwhere(ok[a, 2] & ~ok[a, 1])
        if len(idx):
            i, j = max(idx, key=lambda ij: abs(THETAS[ij[1]]) + abs(VELOCIDADES[ij[0]]) / 2)
            elegido = (N, i, j)
            break
    if elegido is None:
        print("  no hay estados que V_psi recupere y x'Px no; salteo 6_trayectorias")
        return
    N, i, j = elegido
    x0 = [0.0, VELOCIDADES[i], THETAS[j], 0.0]
    p = ParametrosPlanta()
    dt = p.dt
    fig, ejes = plt.subplots(3, 1, figsize=(9, 6.4), sharex=True)
    for tipo in ("x'Px", "V_psi"):
        mpc, env = _mpc(N, tipo, ruta_modelo)
        okk, _, _, tray = lazo_cerrado(mpc, env, x0, guardar=True)
        X = np.array([s for s, _ in tray]); F = np.array([f for _, f in tray])
        tt = np.arange(len(X)) * dt
        lab = ETIQUETA[tipo] + ("  (recupera)" if okk else "  (se cae / sale del riel)")
        ejes[0].plot(tt, X[:, 2], color=COLOR[tipo], label=lab)
        ejes[1].plot(tt, X[:, 0], color=COLOR[tipo])
        ejes[2].step(tt, F, where="post", color=COLOR[tipo])
    ejes[0].axhline(np.deg2rad(46), color=TINTA2, lw=1, ls="--"); ejes[0].axhline(-np.deg2rad(46), color=TINTA2, lw=1, ls="--")
    ejes[1].axhline(p.x_max, color=TINTA2, lw=1, ls="--"); ejes[1].axhline(-p.x_max, color=TINTA2, lw=1, ls="--")
    ejes[2].axhline(p.fuerza_max, color=TINTA2, lw=1, ls="--"); ejes[2].axhline(-p.fuerza_max, color=TINTA2, lw=1, ls="--")
    ejes[0].set_ylabel(r"$\theta$ [rad]"); ejes[1].set_ylabel(r"$x_c$ [m]"); ejes[2].set_ylabel("F [N]")
    ejes[2].set_xlabel("t [s]")
    ejes[0].legend(loc="upper right")
    ejes[0].set_title(f"Lazo cerrado desde θ₀ = {THETAS[j]:.2f} rad, ẋ₀ = {VELOCIDADES[i]:.2f} m/s, N = {N}  "
                      "(punteado: límites)")
    fig.tight_layout()
    fig.savefig(salida)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default=str(AQUI / "modelos" / "V_psi_N5_s0.pt"))
    ap.add_argument("--porcion", nargs=3, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    ruta = Path(a.modelo).resolve()
    if a.porcion is not None:
        k, n, salida = a.porcion
        correr_porcion(int(k), int(n), str(ruta), salida)
        sys.exit(0)
    dest = AQUI / "figuras"
    dest.mkdir(exist_ok=True)

    red, cfg, hist = cargar(ruta)
    print("1/6 aprendizaje"); fig_aprendizaje(hist, cfg, dest / "1_aprendizaje.png")
    print("2/6 valor"); fig_valor(red, dest / "2_valor.png")
    print("3/6 bellman"); fig_bellman(red, cfg, dest / "3_bellman.png")
    print("4/6 barrido en lazo cerrado")
    ok, costo, t = barrido(ruta, dest / "barrido.npz")
    fig_regiones(ok, dest / "4_regiones.png")
    print("5/6 horizonte"); fig_horizonte(ok, t, dest / "5_horizonte.png")
    print("6/6 trayectorias"); fig_trayectorias(ok, ruta, dest / "6_trayectorias.png")
    for a_, N in enumerate(HORIZONTES):
        print(f"N={N}: " + "  ".join(f"{tipo} {ok[a_, b].sum()}/{ok[a_, b].size} ({1e3 * np.nanmean(t[a_, b]):.1f} ms)"
                                     for b, tipo in enumerate(TIPOS)))
