"""Animaciones de los estados iniciales que V_psi recupera y x'Px no.

    python Inciso_A/animaciones.py            (requiere figuras/barrido.npz de visualizaciones.py)

Genera en Inciso_A/figuras/:
    casos_vpsi_vs_xpx.json    trayectorias de todos esos casos (x, theta, F por paso), para la página
    anim_N{N}_th{..}_v{..}.gif  carro-péndulo lado a lado para algunos casos representativos
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import visualizaciones as viz  # noqa: E402
from pendulo_env import ParametrosPlanta  # noqa: E402


def casos(ok):
    """(N, i, j) donde V_psi recupera y x'Px no."""
    b_xpx, b_vpsi = viz.TIPOS.index("x'Px"), viz.TIPOS.index("V_psi")
    return [(N, int(i), int(j)) for a, N in enumerate(viz.HORIZONTES)
            for i, j in np.argwhere(ok[a, b_vpsi] & ~ok[a, b_xpx])]


def simular(N, i, j, ruta):
    x0 = [0.0, float(viz.VELOCIDADES[i]), float(viz.THETAS[j]), 0.0]
    caso = {"N": N, "theta0": x0[2], "v0": x0[1], "corridas": {}}
    for tipo in ("x'Px", "V_psi"):
        mpc, env = viz._mpc(N, tipo, ruta)
        ok, costo, _, tray = viz.lazo_cerrado(mpc, env, x0, guardar=True)
        X = np.array([s for s, _ in tray])
        F = np.array([f for _, f in tray])
        motivo = None
        if not ok:
            ult = X[-1]
            motivo = "sale del riel" if abs(ult[0]) > env.p.x_max else (
                "se cae" if abs(ult[2]) > env.p.theta_corte else "no converge en 5 s")
        caso["corridas"][tipo] = {
            "ok": ok, "motivo": motivo, "costo": round(costo, 1),
            "x": np.round(X[:, 0], 4).tolist(), "th": np.round(X[:, 2], 4).tolist(),
            "F": [None if np.isnan(f) else round(float(f), 2) for f in F],
        }
    return caso


def gif(caso, salida, p):
    """Dos carros-péndulo lado a lado, 50 fps (un cuadro por paso de 20 ms)."""
    tipos = ("x'Px", "V_psi")
    largo = max(len(caso["corridas"][t]["x"]) for t in tipos)
    fig, ejes = plt.subplots(1, 2, figsize=(9, 3.4))
    artistas = []
    for ax, tipo in zip(ejes, tipos):
        c = caso["corridas"][tipo]
        ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.25, 0.75); ax.set_aspect("equal")
        ax.axis("off")
        ax.plot([-p.x_max - 0.05, p.x_max + 0.05], [0, 0], color=viz.EJE, lw=3, solid_capstyle="round")
        for borde in (-p.x_max, p.x_max):
            ax.plot([borde, borde], [-0.06, 0.06], color=viz.TINTA2, lw=2)
        carro = Rectangle((0, 0), 0.16, 0.08, fc=viz.COLOR[tipo], ec="none", zorder=3)
        ax.add_patch(carro)
        (palo,) = ax.plot([], [], color=viz.TINTA, lw=3, solid_capstyle="round")
        (masa,) = ax.plot([], [], "o", color=viz.TINTA, ms=9)
        flecha = ax.annotate("", xy=(0, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=viz.TINTA2, lw=1.5))
        estado = ax.text(0, 0.68, "", ha="center", fontsize=11, fontweight="semibold")
        ax.text(0, -0.2, viz.ETIQUETA[tipo] + " como costo terminal", ha="center", color=viz.TINTA2, fontsize=10)
        artistas.append((c, carro, palo, masa, flecha, estado))
    reloj = fig.text(0.5, 0.02, "", ha="center", color=viz.TINTA2, fontsize=9)
    fig.suptitle(f"θ₀ = {caso['theta0']:.2f} rad, ẋ₀ = {caso['v0']:.2f} m/s, horizonte N = {caso['N']}",
                 color=viz.TINTA, fontsize=11)

    def cuadro(k):
        for c, carro, palo, masa, flecha, estado in artistas:
            n = len(c["x"])
            kk = min(k, n - 1)
            x, th = c["x"][kk], c["th"][kk]
            carro.set_xy((x - 0.08, -0.04))
            px, py = x + p.L * np.sin(th), p.L * np.cos(th)
            palo.set_data([x, px], [0.0, py]); masa.set_data([px], [py])
            F = c["F"][kk] if kk < n - 1 else None
            if F is not None:
                flecha.set_position((x, -0.1)); flecha.xy = (x + 0.0025 * F, -0.1)
                flecha.set_visible(True)
            else:
                flecha.set_visible(False)
            if k >= n - 1 and not c["ok"]:
                estado.set_text("✗ " + c["motivo"]); estado.set_color("#d03b3b")
            elif k >= n - 1 and c["ok"]:
                estado.set_text("✓ estabilizado"); estado.set_color("#0ca30c")
            else:
                estado.set_text("")
        reloj.set_text(f"t = {k * p.dt:.2f} s")
        return []

    anim = FuncAnimation(fig, cuadro, frames=range(0, largo + 25), interval=1000 * p.dt)
    anim.save(salida, writer=PillowWriter(fps=int(round(1 / p.dt))), dpi=80)
    plt.close(fig)


if __name__ == "__main__":
    ruta = str(AQUI / "modelos" / "V_psi_N5_s0.pt")
    dest = AQUI / "figuras"
    d = np.load(dest / "barrido.npz")
    lista = casos(d["ok"])
    print(f"{len(lista)} casos donde V_psi recupera y x'Px no")
    datos = []
    for k, (N, i, j) in enumerate(lista):
        datos.append(simular(N, i, j, ruta))
        print(f"  {k + 1}/{len(lista)}  N={N} theta0={viz.THETAS[j]:+.2f} v0={viz.VELOCIDADES[i]:+.2f}  "
              "x'Px: " + str(datos[-1]["corridas"]["x'Px"]["motivo"]), flush=True)
    # Sanity check: lo simulado tiene que coincidir con el barrido
    inconsistentes = [c for c in datos if c["corridas"]["x'Px"]["ok"] or not c["corridas"]["V_psi"]["ok"]]
    print(f"inconsistencias con el barrido: {len(inconsistentes)}")
    p = ParametrosPlanta()
    meta = {"dt": p.dt, "x_max": p.x_max, "L": p.L, "theta_corte": p.theta_corte, "F_max": p.fuerza_max}
    (dest / "casos_vpsi_vs_xpx.json").write_text(json.dumps({"meta": meta, "casos": datos}), encoding="utf-8")

    # GIFs: con N = 1, el caso más lejano de cada tipo de falla, y uno con N = 5
    elegidos = []
    for N in (1, 5):
        del_N = [c for c in datos if c["N"] == N]
        for motivo in ("sale del riel", "se cae"):
            cand = [c for c in del_N if c["corridas"]["x'Px"]["motivo"] == motivo]
            if cand:
                elegidos.append(max(cand, key=lambda c: abs(c["theta0"]) + abs(c["v0"]) / 2))
    for c in elegidos:
        nombre = f"anim_N{c['N']}_th{c['theta0']:+.2f}_v{c['v0']:+.2f}.gif".replace("+", "p").replace("-", "m")
        print("  gif", nombre, flush=True)
        gif(c, dest / nombre, p)
